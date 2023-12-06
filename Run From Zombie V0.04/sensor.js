class Sensor{
    constructor(avatar){
        this.avatar = avatar;
        this.rayCount = 24;
        this.rayLength = 110;
        this.raySpread = 2 * Math.PI;
        
        this.rays = [];
        this.readings = [];
    }
    
    update(mapBorders,zombies){
       
        this.castRays();
        this.readings = [];
        for(let i=0;i<this.rays.length;i++){
            this.readings.push(
                this.getReading(this.rays[i],mapBorders,zombies)
            );
        }
    }
    
    getReading(ray,mapBorders,zombies){ 
        let touches = [];
        for(let i=0;i<mapBorders.length;i++){
            
            const touch = getIntersection(
                ray[0],
                ray[1],
                mapBorders[i],
                mapBorders[(i+1)%mapBorders.length]
            );
            if(touch){
                touches.push(touch);
                
            }
        }
        
        for(let i=0;i<zombies.length;i++){
            const poly = zombies[i].polygon;
            for(let j=0;j<poly.length;j++){
                const value = getIntersection(
                    ray[0],
                    ray[1],
                    poly[j],
                    poly[(j+1)%poly.length]
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
           // const rayAngle = lerp(this.raySpread/2,
           // -this.raySpread/2,this.rayCount==1?0.5:
           // i/(this.rayCount-1))+this.avatar.angle;
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