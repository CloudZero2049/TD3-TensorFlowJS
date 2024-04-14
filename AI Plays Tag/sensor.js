class Sensor{
    constructor(avatar){
        this.avatar = avatar;
        this.rayCount = 45;
        this.rayLength = 500;
        this.raySpread = 2 * Math.PI;
        this.rays = [];
        this.readings = [];
    }
    update(mapBorders,entities,role){
       
        this.castRays();
        this.readings = [];
        for(let i = 0; i < this.rays.length; i++){
            this.readings.push(
                this.getReading(this.rays[i],mapBorders,entities,role)
            );
        }
    }
    getReading(ray,mapBorders,entities,role){ 
        let touches = [];
        if (role == "runner"){
            for(let i = 0; i < mapBorders.length; i++){
            
                const touch = getIntersection(
                    ray[0],
                    ray[1],
                    mapBorders[i],
                    mapBorders[(i+1) % mapBorders.length],
                    "wall"
                );
                if(touch){
                    touches.push(touch);
                }
            }
        }
        for(let i = entities.length -1; i >= 0; i--){
            if (entities[i] === this.avatar) {continue}
            if (entities[i].role == role) {continue}
            const poly = entities[i].polygon;
            for(let j = 0; j < poly.length; j++){
                const value = getIntersection(
                    ray[0],
                    ray[1],
                    poly[j],
                    poly[(j+1) % poly.length],
                    entities[i].role,
                );
                if(value){touches.push(value);}
            }
        }
        if(touches.length == 0){return null;}
        else {
            const offsets = touches.map(e=>e.offset);
            const minOffset = Math.min(...offsets);
            return touches.find(e=>e.offset==minOffset);
        }
    }
    castRays(){
        this.rays = [];
        for(let i=0;i<this.rayCount;i++){
           const rayAngle = (i / this.rayCount) * this.raySpread;
            const start = {x:this.avatar.x + this.avatar.width/2, y:this.avatar.y + this.avatar.height/2};
            const end = {
                x:this.avatar.x - Math.sin(rayAngle)*this.rayLength,
                y:this.avatar.y - Math.cos(rayAngle)*this.rayLength
            };
            this.rays.push([start,end]);
        }
    }
    draw(ctx){
        for(let i=0;i<this.rayCount;i++){
            let end = this.rays[i][1];
            if(this.readings[i]){
                end = this.readings[i];
            }
            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "yellow";
            ctx.moveTo(this.rays[i][0].x,this.rays[i][0].y);
            ctx.lineTo(end.x,end.y);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "black";
            ctx.moveTo(this.rays[i][1].x,this.rays[i][1].y);
            ctx.lineTo(end.x,end.y);
            ctx.stroke();
        }
    }
}