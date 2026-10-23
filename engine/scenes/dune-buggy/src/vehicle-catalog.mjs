/** Shared vehicle identity, dimensions and kinematic preview tuning. */
export const vehicles = [
 {id:'buggy',name:'Sand rail / 02',code:'SR–02',kind:'Open-frame buggy',description:'Exposed frame · Independent suspension',color:'#344a4b',parameters:{wheelbase:2.62,track:1.88,tireRadius:.395,cageHeight:1.78},handling:{speed:6,reverse:2,acceleration:3.2,braking:9,steering:25,compression:.12,extension:.1,stiffness:45,damping:.85}},
 {id:'trophy',name:'Baja / T1',code:'T–01',kind:'Trophy truck',description:'Long travel · Open bed · Panel bodywork',color:'#c95027',parameters:{wheelbase:3,track:2.1,tireRadius:.44,cageHeight:1.84},handling:{speed:7.5,reverse:2,acceleration:3.5,braking:8,steering:25,compression:.15,extension:.1,stiffness:36,damping:.8}},
 {id:'rally',name:'Vector / R3',code:'R–03',kind:'Rally hatchback',description:'Compact chassis · Enclosed cabin · Rally aero',color:'#1c487b',parameters:{wheelbase:2.3,track:1.72,tireRadius:.34,cageHeight:1.62},handling:{speed:8,reverse:2.4,acceleration:4.2,braking:10,steering:30,compression:.08,extension:.05,stiffness:70,damping:.95}},
 {id:'monster',name:'Goliath / M4',code:'M–04',kind:'Monster truck',description:'Oversized tires · Lifted chassis · Wide stance',color:'#738c24',parameters:{wheelbase:3,track:2.5,tireRadius:.72,cageHeight:2.16},handling:{speed:4.5,reverse:1.6,acceleration:2.1,braking:7,steering:22,compression:.16,extension:.12,stiffness:28,damping:.7}},
 {id:'derby',name:'Brawler / D8',code:'D–08',kind:'Derby sedan',description:'Reinforced doors · Trunk body · Window guards',color:'#964d55',parameters:{wheelbase:2.9,track:1.96,tireRadius:.36,cageHeight:1.64},handling:{speed:6,reverse:2.5,acceleration:3.4,braking:8,steering:27,compression:.08,extension:.06,stiffness:65,damping:1}},
 {id:'sprint',name:'Firefly / S9',code:'S–09',kind:'Sprint car',description:'Open wheels · High wing · Exposed cage',color:'#d4a12d',parameters:{wheelbase:2.4,track:1.85,tireRadius:.37,cageHeight:1.65},handling:{speed:9,reverse:1.5,acceleration:4.8,braking:10,steering:25,compression:.07,extension:.05,stiffness:75,damping:.9}},
 {id:'semi',name:'Hauler / H6',code:'H–06',kind:'Semi + trailer',description:'Tall cab · Fifth wheel · Articulated flatbed',color:'#347b83',parameters:{wheelbase:3,track:2.1,tireRadius:.44,cageHeight:2.25},handling:{speed:4.5,reverse:1.2,acceleration:1.8,braking:6,steering:24,compression:.10,extension:.08,stiffness:55,damping:1.1}},
];
export const vehicleById=id=>vehicles.find(v=>v.id===(id??'buggy'));
export const vehicleLift=p=>p.vehicle==='monster'?p.tireRadius-.395:0;
export const tireWidthScale=p=>p.vehicle==='monster'?1.5:1;
export function vehicleFields(id){
 const monster=id==='monster',rally=id==='rally',semi=id==='semi';
 return [['wheelbase','Wheelbase',rally?2.3:2.3,monster?3.4:3,.02,'m'],['track','Track width',monster?2.3:1.65,monster?2.7:2.1,.01,'m'],['tireRadius','Tire radius',monster?.62:rally?.32:.34,monster?.82:.46,.005,'m'],['cageHeight','Cage height',monster?2.05:semi?2.05:1.6,monster||semi?2.4:1.98,.02,'m']];
}
export function mechanicalParameters(parameters){
 const lift=vehicleLift(parameters);
 return {...parameters,vehicle:'buggy',tireRadius:parameters.tireRadius-lift,cageHeight:parameters.cageHeight-lift};
}

export const trailerSpec=p=>p.vehicle==='semi'?{hitch:[0,1.12,p.wheelbase/2+.8],axleZ:p.wheelbase/2+3.75,wheelRadius:.44,halfTrack:1.04,maxYawRad:.95}:null;
