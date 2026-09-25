import json, math, os, re
from pathlib import Path
import pymupdf

PDF=Path("02_Изометрии_10_листов.pdf")
PAGE_INDEX=int(os.getenv("DIMENSION_PAGE_INDEX", "1"))
PAGE_TAG=PAGE_INDEX+1
MAX_WIDTH_PT=.72; MIN_LENGTH_PT=20; ARROW_DISTANCE_PT=16
BLUE_MIN_WIDTH_PT=.70
BASE_ENDPOINT_ARROW_DISTANCE_PT=6
MIN_TEXT_OFFSET_PT=3; MAX_TEXT_OFFSET_PT=24; MAX_TEXT_ALONG_MARGIN_PT=12

def diff(a,b): return abs((a-b+90)%180-90)

doc=pymupdf.open(PDF); page=doc[PAGE_INDEX]
numbers=[]
service_boxes=[]
candidate_index=0
for block in page.get_text("dict").get("blocks",[]):
    for line in block.get("lines",[]):
        full=" ".join(s.get("text","") for s in line.get("spans",[])).upper()
        service_line=any(t in full for t in ("Z+","DN"," X "," Y "," Z "))
        if re.search(r"\bDN\s*\d+(?:\s*[XХ×]\s*\d+)+\b", full):
            line_spans=line.get("spans",[])
            if line_spans:
                boxes=[s["bbox"] for s in line_spans]
                service_boxes.append({
                    "text":full,
                    "bbox":[min(b[0] for b in boxes),min(b[1] for b in boxes),
                            max(b[2] for b in boxes),max(b[3] for b in boxes)]
                })
        dx,dy=line.get("dir",(1,0)); ta=math.degrees(math.atan2(dy,dx))%180
        for span in line.get("spans",[]):
            text=span.get("text","").strip()
            if not text.isdigit() or int(text)<1: continue
            candidate_index += 1
            x0,y0,x1,y1=span["bbox"]
            if not service_line:
                numbers.append({"candidate_id":f"C{candidate_index}","text":text,"x":(x0+x1)/2,"y":(y0+y1)/2,
                                "x0":x0,"y0":y0,"x1":x1,"y1":y1,"angle":ta})

# Boxed integers are item / position labels, not linear dimensions.
frame_rects=[]
for drawing in page.get_drawings():
    for item in drawing.get("items",[]):
        if item[0]=="re":
            rect=item[1]
            if 4<=rect.width<=40 and 4<=rect.height<=40:
                frame_rects.append(rect)
numbers=[n for n in numbers if not any(
    rect.x0<=n["x"]<=rect.x1 and rect.y0<=n["y"]<=rect.y1
    for rect in frame_rects
)]

arrows=[]
arrowheads=[]
for d in page.get_drawings():
    r=d.get("rect")
    if d.get("type")=="f" and r and 2<=r.width<=20 and 2<=r.height<=20:
        arrows.append(((r.x0+r.x1)/2,(r.y0+r.y1)/2))
        # A triangular filled path is an arrowhead.  The vertex opposite the
        # shortest edge is its tip, so its direction can be recovered exactly.
        points=[]
        for item in d.get("items",[]):
            if item[0]=="l":
                for pt in (item[1],item[2]):
                    if not any(math.hypot(pt.x-u.x,pt.y-u.y)<.1 for u in points):
                        points.append(pt)
        if len(points)==3:
            pairs=[(math.hypot(points[i].x-points[j].x,points[i].y-points[j].y),i,j)
                   for i,j in ((0,1),(0,2),(1,2))]
            _,i,j=min(pairs)
            k=3-i-j
            base=((points[i].x+points[j].x)/2,(points[i].y+points[j].y)/2)
            arrowheads.append({"tip":(points[k].x,points[k].y),"base":base})

service_leaders=[]
thick_segments=[]
for drawing in page.get_drawings():
    width=float(drawing.get('width') or 0)
    if width>MAX_WIDTH_PT+.1:
        for item in drawing.get('items',[]):
            if item[0]=='l': thick_segments.append((item[1],item[2]))
def point_segment_distance(x,y,a,b):
    vx,vy=b.x-a.x,b.y-a.y; den=vx*vx+vy*vy
    if den==0: return math.hypot(x-a.x,y-a.y)
    t=max(0,min(1,((x-a.x)*vx+(y-a.y)*vy)/den))
    return math.hypot(x-(a.x+t*vx),y-(a.y+t*vy))
for service in service_boxes:
    x0,y0,x1,y1=service["bbox"]
    best=None
    for drawing in page.get_drawings():
        width=float(drawing.get("width") or 0)
        # PDF stores nominal 0.72 pt lines as 0.720000028... sometimes.
        if not BLUE_MIN_WIDTH_PT<=width<=MAX_WIDTH_PT+.02: continue
        for item in drawing.get("items",[]):
            if item[0]!="l": continue
            p,q=item[1],item[2]
            length=math.hypot(q.x-p.x,q.y-p.y)
            if not 8<=length<=120: continue
            def rect_distance(x,y):
                dx=max(x0-x,0,x-x1); dy=max(y0-y,0,y-y1)
                return math.hypot(dx,dy)
            dp=rect_distance(p.x,p.y); dq=rect_distance(q.x,q.y)
            near=min(dp,dq)
            if near>16: continue
            text_end,axis_end=(p,q) if dp<=dq else (q,p)
            lvx,lvy=axis_end.x-text_end.x,axis_end.y-text_end.y
            llen=math.hypot(lvx,lvy)
            valid=[]
            for arrow in arrowheads:
                tx,ty=arrow["tip"]; bx,by=arrow["base"]
                tip_distance=math.hypot(axis_end.x-tx,axis_end.y-ty)
                if tip_distance>8: continue
                avx,avy=tx-bx,ty-by
                alen=math.hypot(avx,avy)
                if not alen or (lvx*avx+lvy*avy)/(llen*alen)<.75: continue
                axis_distance=min(point_segment_distance(tx,ty,a,b)
                                  for a,b in thick_segments) if thick_segments else 999
                if axis_distance>5: continue
                valid.append((tip_distance+axis_distance,arrow))
            if not valid: continue
            score=near+min(valid)[0]
            if best is None or score<best[0]: best=(score,p,q)
    if best is not None: service_leaders.append(best[1:])

matches=[]
for drawing in page.get_drawings():
    width=float(drawing.get("width") or 0)
    if not 0<width<=MAX_WIDTH_PT: continue
    for item in drawing.get("items",[]):
        if item[0]!="l": continue
        a,b=item[1],item[2]; vx,vy=b.x-a.x,b.y-a.y; length=math.hypot(vx,vy)
        if length<MIN_LENGTH_PT: continue
        ux,uy=vx/length,vy/length; la=math.degrees(math.atan2(vy,vx))%180
        if min(diff(la,x) for x in (0,30,90,150))>8: continue
        aa=min((math.hypot(a.x-x,a.y-y),i) for i,(x,y) in enumerate(arrows))
        ab=min((math.hypot(b.x-x,b.y-y),i) for i,(x,y) in enumerate(arrows))
        if aa[0]>ARROW_DISTANCE_PT or ab[0]>ARROW_DISTANCE_PT or aa[1]==ab[1]: continue
        mx,my=(a.x+b.x)/2,(a.y+b.y)/2; best=None
        for n in numbers:
            rx,ry=n["x"]-a.x,n["y"]-a.y
            along_signed=rx*ux+ry*uy
            perpendicular=abs(rx*(-uy)+ry*ux)
            if (along_signed < -MAX_TEXT_ALONG_MARGIN_PT or
                    along_signed > length+MAX_TEXT_ALONG_MARGIN_PT or
                    not MIN_TEXT_OFFSET_PT<=perpendicular<=MAX_TEXT_OFFSET_PT): continue
            if diff(n["angle"],la)>12: continue
            along_distance=max(0,-along_signed,along_signed-length)
            score=perpendicular+along_distance
            if best is None or score<best[0]: best=(score,n,perpendicular,along_distance)
        if best:
            matches.append({"mode":"centered","candidate_id":best[1]["candidate_id"],"value":int(best[1]["text"]),"a":[a.x,a.y],"b":[b.x,b.y],"width":width,
                            "length":round(length,2),"angle":round(la,2),
                            "text_offset":round(best[2],2),"text_along":round(best[3],2)})

# Separate rule for a remote value connected by a thin leader to an arrow cluster.
leader_candidates={}
for drawing in page.get_drawings():
    width=float(drawing.get("width") or 0)
    if not 0<width<=MAX_WIDTH_PT: continue
    for item in drawing.get("items",[]):
        if item[0]!="l": continue
        p,q=item[1],item[2]; leader_length=math.hypot(q.x-p.x,q.y-p.y)
        if not 8<=leader_length<=120: continue
        for joint,outer in ((p,q),(q,p)):
            close_arrows=[(x,y) for x,y in arrows if math.hypot(joint.x-x,joint.y-y)<=35]
            if len(close_arrows)<2: continue
            nearest=min(numbers,key=lambda n:math.hypot(outer.x-n["x"],outer.y-n["y"]))
            number_distance=math.hypot(outer.x-nearest["x"],outer.y-nearest["y"])
            if number_distance>12: continue
            value=int(nearest["text"])
            base=None
            leader_angle=math.degrees(math.atan2(q.y-p.y,q.x-p.x))%180
            for bd in page.get_drawings():
                if not 0<float(bd.get("width") or 0)<=MAX_WIDTH_PT: continue
                for bi in bd.get("items",[]):
                    if bi[0]!="l": continue
                    u,v=bi[1],bi[2]; base_length=math.hypot(v.x-u.x,v.y-u.y)
                    if not 8<=base_length<=120: continue
                    # The target must be a real dimension segment: one distinct
                    # arrowhead near each endpoint. This prevents a leader from
                    # attaching to an arbitrary nearby construction line.
                    arrow_u=min((math.hypot(u.x-x,u.y-y),i) for i,(x,y) in enumerate(arrows))
                    arrow_v=min((math.hypot(v.x-x,v.y-y),i) for i,(x,y) in enumerate(arrows))
                    if (arrow_u[0]>BASE_ENDPOINT_ARROW_DISTANCE_PT or
                            arrow_v[0]>BASE_ENDPOINT_ARROW_DISTANCE_PT or
                            arrow_u[1]==arrow_v[1]):
                        continue
                    center_distance=math.hypot((u.x+v.x)/2-joint.x,(u.y+v.y)/2-joint.y)
                    base_angle=math.degrees(math.atan2(v.y-u.y,v.x-u.x))%180
                    if center_distance>5 or diff(base_angle,leader_angle)<15: continue
                    score=center_distance-base_length*.01
                    if base is None or score<base[0]: base=(score,u,v)
            if base is None: continue
            key=(nearest["text"],round(nearest["x"],2),round(nearest["y"],2))
            total_score=number_distance+base[0]
            candidate={"mode":"leader","candidate_id":nearest["candidate_id"],"value":value,"a":[base[1].x,base[1].y],"b":[base[2].x,base[2].y],
                       "leader_a":[p.x,p.y],"leader_b":[q.x,q.y],"width":width,
                       "length":round(leader_length,2),"angle":None}
            if key not in leader_candidates or total_score<leader_candidates[key][0]:
                leader_candidates[key]=(total_score,candidate)

matches.extend(item[1] for item in leader_candidates.values())

def point_distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def segment_contains(outer, inner):
    ax, ay = outer[1][0]-outer[0][0], outer[1][1]-outer[0][1]
    length_sq = ax*ax + ay*ay
    if not length_sq:
        return False
    projections=[]; cross=[]
    for x,y in inner:
        projections.append(((x-outer[0][0])*ax + (y-outer[0][1])*ay)/length_sq)
        cross.append(abs(ax*(y-outer[0][1])-ay*(x-outer[0][0]))/math.sqrt(length_sq))
    return max(cross) <= 3 and min(projections) >= -.02 and max(projections) <= 1.02

auto_nested={}
for inner in matches:
    inner_line=[inner["a"],inner["b"]]
    inner_len=point_distance(*inner_line)
    for outer in matches:
        if inner is outer:
            continue
        outer_line=[outer["a"],outer["b"]]
        if point_distance(*outer_line) > inner_len*1.05 and segment_contains(outer_line,inner_line):
            auto_nested[inner["candidate_id"]]=outer["candidate_id"]
            inner["nested"]=True
            inner["covers"]=outer["candidate_id"]
            break
for item in matches:
    item.setdefault("nested",False)

output_dir=Path("output")/"dimension_probe"
output_dir.mkdir(parents=True,exist_ok=True)
out=output_dir/f"dimension_probe_page{PAGE_TAG}.json"
out.write_text(json.dumps(matches,ensure_ascii=False,indent=2),encoding="utf-8")
for i,m in enumerate(matches,1):
    a,b=pymupdf.Point(*m["a"]),pymupdf.Point(*m["b"])
    color=(0.55,0.18,0.72) if m.get("nested") else (1,0,0)
    page.draw_line(a,b,color=color,width=1.4 if m.get("nested") else 1,overlay=True)
    if m["mode"]=="leader":
        page.draw_line(pymupdf.Point(*m["leader_a"]),pymupdf.Point(*m["leader_b"]),color=(0,0.7,0),width=1,overlay=True)
    source_number=next((n for n in numbers if n["candidate_id"]==m["candidate_id"]),None)
    if source_number is not None:
        box=pymupdf.Rect(source_number["x0"]-2,source_number["y0"]-2,
                         source_number["x1"]+2,source_number["y1"]+2)
        page.draw_rect(box,color=color,width=1.2,overlay=True)
for service in service_boxes:
    x0,y0,x1,y1=service["bbox"]
    page.draw_rect(pymupdf.Rect(x0-2,y0-2,x1+2,y1+2),color=(0,0.35,1),width=1,overlay=True)
for p,q in service_leaders:
    page.draw_line(p,q,color=(0,0.35,1),width=1,overlay=True)
page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(output_dir/f"dimension_probe_page{PAGE_TAG}.png")
print(f"saved {out}: {len(matches)} strict dimension lines")
