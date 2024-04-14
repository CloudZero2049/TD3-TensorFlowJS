
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
   
    points.push({ x: this.x, y: this.y});
    points.push({ x: this.x + this.width, y: this.y});
    points.push({ x: this.x + this.width, y: this.y + this.height});
    points.push({ x: this.x, y: this.y + this.height});
    this.polygon = points;
    return points;
}
  draw(){
    ctx.beginPath();
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x,this.y,this.width,this.height);
    ctx.fillStyle = "black";
    ctx.font = "bold 20px Arial";
    ctx.fillText(`${this.uid}`, this.x + 3, this.y + this.height, 17);
  }
}
class Humanoid extends Entity{
  constructor(uid,x,y) {
    super(uid,x,y);
    //this.inventory = [];
  }
}
class Player extends Humanoid{
  constructor(uid,x,y,color,role) {
    super(uid,x,y);
    this.type = "player";
    this.color = color;
    this.role = role; // "runner" or "chaser"
    this.hp = 100;
    this.width = 20;
    this.height = 20;
    this.stunned = false;
    this.speed = 5;
    this.speedMult = 1;
    this.tags = 0;
    this.runTime = 0;
    this.sensor = new Sensor(this);
    this.action = [];
    this.location = [this.x + (this.width/2), this.y + (this.height/2)];
    this.defaults = [[]];
    this.stateSpace = [[]];
    this.nextStateSpace = [[]];
  }
  updateRunnerSensor(reset) {
    const os = observationSpace;
    this.sensor.update(Game.mapBorders,Game.entities,"runner");

    const offsetsXY = this.sensor.readings.map((reading) => {
      if (reading == null) {
          return null;  // Default value for no detection
      } else { 
          
          if (reading.ent === "chaser") {
              
              return {xy:[reading.x,reading.y],type:"chaser"};

          } else if (reading.ent === "wall") { 

              return {xy:[reading.x,reading.y],type:"wall"};
              
          } else {
              // Default value for unknown types
              return null;
          }
      }
    }); // end get offsetsXy

    for (let i = 0; i < offsetsXY.length; i++){
      let distance;
      
      if (offsetsXY[i] == null) {distance = 1}
      else {
        distance = utilsAI.distance(this.location[0], this.location[1], offsetsXY[i].xy[0], offsetsXY[i].xy[1]);
        distance *= os.xyNorm; //scaled (0,1)
        if (offsetsXY[i].type == "chaser") {distance = -(1 - distance);} // (-1,0) descending for zombie
        else {distance = 1 - distance} //(0,1) ascending for walls
        
        distance = Math.floor(distance * agent.decScale);
        distance /=  agent.decScale;
        
      }
      if (!reset) {this.nextStateSpace[0][i+2] = distance;}
      else {this.defaults[0][i+2] = distance;}
      
    } // end for loop
  }
  updateChaserSensor(reset) {
    this.sensor.update(Game.mapBorders,Game.entities,"chaser");

    const offsets = this.sensor.readings.map((reading) => {
      if (reading == null) {
          return 1;  // Default value for no detection
      } else {
          // Adjust the offset based on the type of the detected object
          if (reading.ent === "runner") {
              // Map distances from 1 to 0 (decending)
              
              return parseFloat(reading.offset.toFixed(4));
          } else if (reading.type === "wall") { // disabled
              // Map distances from 0 to -1 (decending)
            
          } else {
              return parseFloat(reading.offset.toFixed(4)); // Default value for unknown types
          }
      }
  });

  for (let i = 0; i < offsets.length; i++){
    let copy = JSON.parse(JSON.stringify(offsets[i]));
    const clipCopy = parseFloat((copy).toFixed(4));
    
    if (!reset) {this.nextStateSpace[0][i+2] = clipCopy;}
    else {this.defaults[0][i+2] = 1;}
    }
  } // end chaser sensor
}

