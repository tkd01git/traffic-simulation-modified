/**
##########################################################
stationary detector  object constructor:
##########################################################
creates a stationary (cross-section) detector 
at a road segment at logical longitudinal coordinate u, 
updating its counts every dtAggr seconds. 

The macroscopic output also aggregates over all lanes while the 
microscopic output also gives the lane index of the passage event

@param road:          the road segement at which the detector is positioned
@param u:             the logical longitudinal coordinate [m] of the detector
@param dtAggr:        aggregation time [s] of the macroscopic output
*/

function stationaryDetector(road,u,dtAggr){
  //console.log("in stationaryDetector cstr: road=",road);
this.road=road;
this.u=u;
this.dtAggr=dtAggr;
this.exportString=""; // for export to file

  if(this.u>road.roadLen){
console.log("Warning: trying to place a detector at position u=",
      this.u," greater than the road segment length ",
      road.roadLen," resetting u=",road.roadLen);
this.u=road.roadLen;
  }

  // initializing macroscopic records

  this.iAggr=0;
  this.historyFlow=[];
  this.historySpeed=[];
  this.historyFlow[0]=0;
  this.historySpeed[0]=0;
  this.vehCount=0; // counting inside each aggregation interval (all lanes)
  this.speedSum=0; // summing inside each aggregation interval
  this.nLanes=this.road.nLanes;
  this.vehNearOld=(this.u<0.5*this.road.roadLen) 
? this.road.findLeaderAt(this.u) : this.road.findFollowerAt(this.u);
}

stationaryDetector.prototype.getAverageSpeedInRange = function (road, uMin, uMax) {
  // 範囲内の車両をフィルタリング
  const vehiclesInRange = road.veh.filter(vehicle => {
    return vehicle.u >= uMin && vehicle.u <= uMax;
  });

  // 範囲内に車両がいない場合は null を返す
  if (vehiclesInRange.length === 0) {
    console.log("指定範囲に車両が存在しません。");
    return null;
  }

  // 範囲内の車両から速度情報を取得し、km/h に変換
  const speeds = vehiclesInRange.map(vehicle => vehicle.speed * 3.6); // m/s → km/h
  // 平均速度を計算
  const averageSpeed = speeds.reduce((sum, speed) => sum + speed, 0) / speeds.length;

  return averageSpeed;
};





stationaryDetector.prototype.update = function(time, dt) {
  var vehNear = (this.u < 0.5 * this.road.roadLen)
      ? this.road.findLeaderAt(this.u) 
      : this.road.findFollowerAt(this.u);

  if (vehNear.id !== this.vehNearOld.id) {
      this.vehNearOld = vehNear;
      this.vehCount++;
      this.speedSum += vehNear.speed;
  }

  if (time >= this.iAggr * this.dtAggr + this.dtAggr) {
      this.iAggr++;
      this.historyFlow[this.iAggr] = this.vehCount / this.dtAggr;
      this.historySpeed[this.iAggr] = (this.vehCount > 0) ? (this.speedSum / this.vehCount) : 0;
      this.vehCount = 0;
      this.speedSum = 0;

      if (downloadActive) {
          this.updateExportString();
      }
  }
};


// updateExportStringの修正版
stationaryDetector.prototype.updateExportString = function () {
  var rest = time / this.dtAggr - Math.floor((time + 0.01) / this.dtAggr);

  if (rest < dt - 0.01) {
    var flowStr = Math.round(3600 * this.historyFlow[this.iAggr]);
    var speedStr = (this.historyFlow[this.iAggr] > 0)
        ? Math.round(3.6 * this.historySpeed[this.iAggr])
        : "--";

    // 指定区間の最低速度を計算
    let averageSpeed = this.getAverageSpeedInRange(this.road, 5100, 5500);
    var numStr = (averageSpeed !== null) ? Math.round(averageSpeed) : "--";

    // exportStringへの追加
    this.exportString = this.exportString + "\n" + time.toFixed(0)
        + "\t\t" + flowStr + "\t\t" + speedStr + "\t\t" + numStr;
  }
};



stationaryDetector.prototype.writeToFile = function(filename) {
  console.log("\nin road.writeVehiclesSimpleToFile(): roadID=", this.road.roadID,
              " filename=", filename);
  console.log("stationaryDetector.exportString=\n", this.exportString);
  download(this.exportString, filename); // コンソール用出力

  // Excelファイルにも書き込むために、writeToExcelFile関数を呼び出す
  this.writeToExcelFile('data01.xlsx'); // ファイル名を固定
};



stationaryDetector.prototype.reset=function(){
this.iAggr=0;
this.historyFlow=[];
this.historySpeed=[];
this.historyFlow[0]=0;
this.historySpeed[0]=0;
this.vehCount=0; // counting inside each aggregation interval (all lanes)
this.speedSum=0; // summing inside each aggregation interval
}


stationaryDetector.prototype.display=function(textsize){
  //console.log("in stationaryDetector.display(textsize)");

ctx.font=textsize+'px Arial';

var flowStr="Flow: "+Math.round(3600*this.historyFlow[this.iAggr])
  +" veh/h";
var speedStr="Speed: "+((this.historyFlow[this.iAggr]>0)
      ? Math.round(3.6*this.historySpeed[this.iAggr])
      : "--")+" km/h";
var densStr="Dens.: "+((this.historyFlow[this.iAggr]>0)
     ? Math.round(1000*this.historyFlow[this.iAggr]
            /this.historySpeed[this.iAggr])
     : "--")+" veh/km";



  var phi=this.road.get_phi(this.u,this.road.traj);
  var cphi=Math.cos(phi);
  var sphi=Math.sin(phi);
  
  //var toRight_axis=-1.1*this.road.nLanes*this.road.laneWidth;
  var roadWidth=this.road.nLanes*this.road.laneWidth;
  //var toRight_axis=-(1.1+0.8*Math.abs(sphi))*roadWidth;
  var toRight_axis=-0.5*roadWidth-(2.2+1.8*Math.abs(sphi))*laneWidth;
  var xPixCenter=this.road.get_xPix(this.u, toRight_axis);
  var yPixCenter=this.road.get_yPix(this.u, toRight_axis);
  var boxWidth=8.2*textsize;
  var boxHeight=3.6*textsize;

  // the detector line

  var detLineWidth=0.2;    // [m]
  var detLineDist=2;     // dist of the loops of double-loop detector [m]
  var detLineLength=this.road.nLanes*this.road.laneWidth;

  var xCenterPix1=  scale*this.road.traj[0](this.u-0.5*detLineDist);
var yCenterPix1= -scale*this.road.traj[1](this.u-0.5*detLineDist);//minus!!
  var xCenterPix2=  scale*this.road.traj[0](this.u+0.5*detLineDist);
  var yCenterPix2= -scale*this.road.traj[1](this.u+0.5*detLineDist); 
  var wPix=scale*detLineWidth;
  var lPix=scale*detLineLength;

  ctx.fillStyle="rgb(0,0,0)";
  ctx.setTransform(cphi,-sphi,sphi,cphi,xCenterPix1,yCenterPix1);
  ctx.fillRect(-0.5*wPix, -0.5*lPix, wPix, lPix); // first line double det
  ctx.setTransform(cphi,-sphi,sphi,cphi,xCenterPix2,yCenterPix2);
  ctx.fillRect(-0.5*wPix, -0.5*lPix, wPix, lPix); // second line double det


  // the textbox

  ctx.setTransform(1,0,0,1,0,0); 
  ctx.fillStyle="rgb(255,255,255)";
  ctx.fillRect(xPixCenter-0.5*boxWidth, yPixCenter-0.5*boxHeight,boxWidth,boxHeight);
  ctx.fillStyle="rgb(0,0,0)";
  ctx.fillText(flowStr,xPixCenter-0.46*boxWidth,yPixCenter-0.2*boxHeight);
  ctx.fillText(speedStr,xPixCenter-0.46*boxWidth,yPixCenter+0.1*boxHeight);
  ctx.fillText(densStr,xPixCenter-0.46*boxWidth,yPixCenter+0.4*boxHeight);
}
