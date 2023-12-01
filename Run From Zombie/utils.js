function getNearestPoint(loc, points, threshhold = Number.MAX_SAFE_INTEGER) {
    let minDist = Number.MAX_SAFE_INTEGER;
    let nearest = null;
    for (const point of points) {
        const dist = distance(point, loc);
        if (dist < minDist && dist < threshhold){
            minDist = dist;
            nearest = point;
        }
    }
    return nearest;
}

function distance(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function average(p1, p2) {
    return new Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

function dot(p1, p2) {
    return p1.x * p2.x + p1.y * p2.y;
}

function add(p1, p2) {
    return new Point(p1.x + p2.x, p1.y + p2.y);
}

function subtract(p1, p2) {
    return new Point(p1.x - p2.x, p1.y - p2.y);
}

function scale(p, scaler) {
    return new Point(p.x * scaler, p.y * scaler);
}

function normalize(p) {
    return scale(p, 1 / magnitude(p));
}

function magnitude(p) {
    return Math.hypot(p.x, p.y);
}

function translate(loc, angle, offset) {
    return new Point(
        loc.x + Math.cos(angle) * offset,
        loc.y + Math.sin(angle) * offset
    );
}

function angle(p) {
    return Math.atan2(p.y, p.x);
}

function findAngle(from,to) {
    let fX = from.x;
    let fY = from.y;
    let tX = to.x;
    let tY = to.y;
  
    let dx = tX - fX;
    let dy = tY - fY;
  
    let angle = Math.atan2(dx, dy);
    return angle;
  }

  function findArrayAngle(from,to) { // [x,y], [x,y]
    let fX = from[0];
    let fY = from[1];
    let tX = to[0];
    let tY = to[1];
    let dx = tX - fX;
    let dy = tY - fY;
  
    let angle = Math.atan2(dy, dx);
    
    return angle;
}

function getIntersection(A,B,C,D){
    /*
    
    Ix = Ax+(Bx-Ax)t = Cx+(Dx-Cx)u;
    Iy = Ay+(By-Ay)t = Cy+(Dy-Cy)u;
    
    Ax+(Bx-Ax)t = Cx+(Dx-Cx)u | -Cx
    (Ax-Cx)+(Bx-Ax)t = (Dx-Cx)u
    
    Ay+(By-Ay)t = Cy+(Dy-Cy)u | -Cy
    (Ay-Cy)+(By-Ay)t = (Dy-Cy)u |*(Dx-Cx)
    
    (Dx-Cx)(Ay-Cy)+(Dx-Cx)(By-Ay)t = 
    = (Dy-Cy)(Ax-Cx)+(Dy-Cy)(Bx-Ax)t |-(Dy-Cy)(Ax-Cx)
                                     |-(Dx-Cx)(By-Ay)t
    */    
        const tTop = (D.x-C.x)*(A.y-C.y)-(D.y-C.y)*(A.x-C.x);
        const uTop = (C.y-A.y)*(A.x-B.x)-(C.x-A.x)*(A.y-B.y);
        const bottom = (D.y-C.y)*(B.x-A.x)-(D.x-C.x)*(B.y-A.y);
       // const t = top/bottom;
        
        const eps = 0.001;
        if(Math.abs(bottom) > eps){        // Modified because float numbers close to zero
            const t = tTop / bottom;
            const u = uTop / bottom;
            if(t >= 0 && t <= 1 && u >= 0 && u <= 1){
                return {
                    x: lerp(A.x, B.x, t),
                    y: lerp(A.y, B.y, t),
                   // bottom:bottom,
                    offset: t,
                }
            }
        }
    
        return null;
        
    }    
        
    function lerp(a,b,t){
        return a + (b-a) * t;
    }

    function getRandomColor() {
        const hue = 290 + Math.random() * 260;
        return "hsl(" + hue + ", 100%, 60%)";
    }
