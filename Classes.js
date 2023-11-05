
class Entity {
  constructor(uid,x,y) {
    this.uid = uid;
    this.x = x;
    this.y = y;
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
    this.speed = 3;
    this.speedMult = 2;
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
  }
}

