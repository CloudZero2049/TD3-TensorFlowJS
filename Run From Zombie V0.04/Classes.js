
class Entity {
  constructor(uid,x,y) {
    this.uid = uid;
    this.x = x;
    this.y = y;
    this.width = 20;
    this.height = 20;
    this.polygon = [];
    this.lastMoveX = 0;
    this.lastMoveY = 0;
    this.speed = 1;
    this.speedMult = 1;
    this.color = "red";
  }
  addPath() {
ctx.moveTo(this.x,this.y);
ctx.lineTo(this.x + this.width,this.y);
ctx.lineTo(this.x + this.width,this.y + this.height);
ctx.lineTo(this.x,this.y + this.height);
ctx.closePath();
  }
  createPolygon(){
    const points = [];
    //const rad = Math.hypot(this.width,this.height)/2;
    //const alpha = Math.atan2(this.width,this.height);
    points.push({ x: this.x, y: this.y
        //x:this.x-Math.sin(this.angle-alpha)*rad,
       // y:this.y-Math.cos(this.angle-alpha)*rad
    });
    points.push({ x: this.x + this.width, y: this.y
        //x:this.x-Math.sin(this.angle+alpha)*rad,
       // y:this.y-Math.cos(this.angle+alpha)*rad
    });
    points.push({ x: this.x + this.width, y: this.y + this.height
        //x:this.x-Math.sin(Math.PI+this.angle-alpha)*rad,
        //y:this.y-Math.cos(Math.PI+this.angle-alpha)*rad
    });
    points.push({ x: this.x, y: this.y + this.height
        //x:this.x-Math.sin(Math.PI+this.angle+alpha)*rad,
        //y:this.y-Math.cos(Math.PI+this.angle+alpha)*rad
    });
    this.polygon = points;
    return points;
}
  draw(){
    ctx.beginPath;
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x,this.y,this.width,this.height);
  }
}

class Humanoid extends Entity{
  constructor(uid,x,y) {
    super(uid,x,y);
    
    //this.inventory = [];

  }
}

class Player extends Humanoid{
  constructor(uid,x,y,user) {
    super(uid,x,y);
    this.user = user;
    this.type = "player";
    this.color = "blue";
    this.hp = 100;
    this.width = 20;
    this.height = 20;
    this.speed = 5;
    this.speedMult = 1;
    this.sensor = new Sensor(this);
    
  }
  
}

class Civilian extends Humanoid{
  constructor(uid,x,y) {
    super(uid,x,y);
    this.type = "civilian";
    this.color = "orange";
    this.hp = 100;
    this.width = 20;
    this.height = 20;
  }
}

class Zombie extends Humanoid{
  constructor(uid,x,y) {
    super(uid,x,y);
    this.type = "zombie";
    this.color = "green";
    this.hp = 100;
    this.width = 20;
    this.height = 20;
    this.speed = 3;
  }
  chase() {
    const x1 = this.x
    const y1 = this.y
    const x2 = observationSpace.agentCoords[0] - (player.width/2);
    const y2 = observationSpace.agentCoords[1] - (player.height/2);

    let angle = utilsAI.angle([x1,y1],[x2,y2]);

    this.lastMoveX = (this.speed * Math.cos(angle));
    this.lastMoveY = (this.speed * Math.sin(angle));
    this.x += this.lastMoveX;
    this.y += this.lastMoveY;
    const newX = this.x + (zom1.width/2);
    const newY = this.y + (zom1.height/2);
    const zomXY = [newX, newY]

    return zomXY;
  }
}

