import * as THREE from './vendor/three.module.js';
import {GLTFLoader} from './vendor/GLTFLoader.js';

const clamp = value => Math.max(0, Math.min(1, Number(value) || 0));
const semantic = name => String(name || '').toLowerCase().replace(/[^a-z0-9]/g, '').replace(/mixamorig|bip001|bone/g, '');
const ALIASES = {
  hips:['hips','pelvis'], spine:['spine','spine1','spine01'], chest:['spine2','spine02','chest'], upperchest:['spine3','spine03','upperchest'],
  neck:['neck','neck01','neck1'], head:['head'], leftshoulder:['leftshoulder','lshoulder','claviclel'], rightshoulder:['rightshoulder','rshoulder','clavicler'],
  leftarm:['leftarm','leftupperarm','lupperarm','upperarml'], rightarm:['rightarm','rightupperarm','rupperarm','upperarmr'],
  leftforearm:['leftforearm','leftlowerarm','lforearm','lowerarml','forearml'], rightforearm:['rightforearm','rightlowerarm','rforearm','lowerarmr','forearmr'],
  lefthand:['lefthand','lhand','handl'], righthand:['righthand','rhand','handr'],
  leftupleg:['leftupleg','leftupperleg','leftthigh','lthigh','thighl'], rightupleg:['rightupleg','rightupperleg','rightthigh','rthigh','thighr'],
  leftleg:['leftleg','leftlowerleg','leftcalf','lcalf','calfl','shinl'], rightleg:['rightleg','rightlowerleg','rightcalf','rcalf','calfr','shinr'],
  leftfoot:['leftfoot','lfoot','footl'], rightfoot:['rightfoot','rfoot','footr']
};
export function avatarBoneKey(name) {
  const s=semantic(name);
  const humanoidFinger=s.match(/^(left|right)(thumb|index|middle|ring|little)(proximal|intermediate|distal)$/);
  if(humanoidFinger)return `${humanoidFinger[2]==='little'?'pinky':humanoidFinger[2]}0${{proximal:1,intermediate:2,distal:3}[humanoidFinger[3]]}${humanoidFinger[1]==='left'?'l':'r'}`;
  const finger=s.match(/^(left|right)(?:hand)?(thumb|index|middle|ring|pinky|little)([123])$/);
  if(finger)return `${finger[2]==='little'?'pinky':finger[2]}0${finger[3]}${finger[1]==='left'?'l':'r'}`;
  return Object.entries(ALIASES).find(([,names])=>names.includes(s))?.[0] || s;
}
const key=avatarBoneKey;
const POSE_CHILD={hips:'spine',spine:'chest',chest:'upperchest',upperchest:'neck',neck:'head',
  leftshoulder:'leftarm',rightshoulder:'rightarm',leftarm:'leftforearm',rightarm:'rightforearm',
  leftforearm:'lefthand',rightforearm:'righthand',lefthand:'middle01l',righthand:'middle01r',
  leftupleg:'leftleg',rightupleg:'rightleg',leftleg:'leftfoot',rightleg:'rightfoot'};
export function poseDirectionChild(bone,positions){
  // Sparse arm-only libraries do not calibrate the chest/clavicle frame.
  // Keep the authored shoulder attachment rather than folding collar skin.
  if(['leftshoulder','rightshoulder'].includes(key(bone.name))&&!positions.has('chest')&&!positions.has('upperchest'))return null;
  const children=bone.children.filter(o=>o.isBone&&positions.has(key(o.name)));
  const preferred=POSE_CHILD[key(bone.name)];
  // A torso branch must follow the spine, never the first shoulder or thigh.
  return children.find(o=>key(o.name)===preferred)||(children.length===1?children[0]:null);
}
function quaternion(value, type='quaternion') {
  if (!value) return null;
  if (type==='rotation6d' && value.length>=6) {
    const x=new THREE.Vector3(value[0],value[1],value[2]).normalize();
    const y=new THREE.Vector3(value[3],value[4],value[5]);y.addScaledVector(x,-y.dot(x)).normalize();
    const z=new THREE.Vector3().crossVectors(x,y).normalize();
    return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x,y,z));
  }
  if (value.length>=4) return new THREE.Quaternion(value[0],value[1],value[2],value[3]).normalize();
  return null;
}
function morphName(name) {return String(name).toLowerCase().replace(/[^a-z0-9]/g,'');}
export function reflectQuaternion(q,signs=[1,1,1]){
  const determinant=signs[0]*signs[1]*signs[2];
  return new THREE.Quaternion(q.x*determinant*signs[0],q.y*determinant*signs[1],q.z*determinant*signs[2],q.w);
}
export function disposeObject(object){
  const geometry=new Set(),materials=new Set(),textures=new Set();
  object.traverse(o=>{
    if(o.geometry)geometry.add(o.geometry);
    for(const material of o.material?(Array.isArray(o.material)?o.material:[o.material]):[]){
      materials.add(material);
      for(const value of Object.values(material))if(value?.isTexture)textures.add(value);
    }
  });
  for(const item of geometry)item.dispose();
  for(const item of materials)item.dispose();
  for(const item of textures)item.dispose();
}
export function aimBoneToward(bone,child,target){
  const origin=bone.getWorldPosition(new THREE.Vector3());
  const current=child.getWorldPosition(new THREE.Vector3()).sub(origin);
  if(target.lengthSq()<1e-10||current.lengthSq()<1e-10)return false;
  const worldDelta=new THREE.Quaternion().setFromUnitVectors(current.normalize(),target.clone().normalize());
  const parentWorld=bone.parent.getWorldQuaternion(new THREE.Quaternion());
  const localDelta=parentWorld.clone().invert().multiply(worldDelta).multiply(parentWorld);
  bone.quaternion.premultiply(localDelta);bone.updateMatrixWorld(true);return true;
}
export function palmFrame(forward,across){
  const y=forward.clone().normalize();
  const x=across.clone().addScaledVector(y,-across.dot(y));
  if(forward.lengthSq()<1e-10||x.lengthSq()<1e-10)return null;
  x.normalize();const z=new THREE.Vector3().crossVectors(x,y).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x,y,z));
}
export function fingerBend(incoming,outgoing,axis,limit=1.45){
  const n=axis.clone().normalize();
  const a=incoming.clone().addScaledVector(n,-incoming.dot(n));
  const b=outgoing.clone().addScaledVector(n,-outgoing.dot(n));
  if(n.lengthSq()<1e-10||a.lengthSq()<1e-10||b.lengthSq()<1e-10)return 0;
  a.normalize();b.normalize();
  return Math.max(-limit,Math.min(limit,Math.atan2(n.dot(new THREE.Vector3().crossVectors(a,b)),a.dot(b))));
}
export function thumbCurl(first,second){
  if(first.lengthSq()<1e-10||second.lengthSq()<1e-10)return 0;
  return Math.min(.55,first.angleTo(second)*.65);
}
export function fitThumb(bone,name,rig,positions,transform){
  const match=name.match(/^thumb0([123])([lr])$/);if(!match)return false;
  const [,segment,side]=match;
  // A thumb's first bone is an opposition joint, not a finger hinge.
  // Preserve the authored opposition: aiming it at an incompatible source
  // metacarpal folds the webbing into the palm even with an angle limit.
  bone.quaternion.copy(rig.bind.get(bone));
  if(segment==='1'){bone.updateMatrixWorld(true);return true;}
  const a=positions.get(`thumb01${side}`),b=positions.get(`thumb02${side}`),c=positions.get(`thumb03${side}`);
  const next=rig.bones.get(`thumb03${side}`),middle=rig.bones.get(`middle01${side}`);
  const proximal=rig.bones.get(`thumb02${side}`);
  if(a&&b&&c&&next&&middle&&proximal){
    const direction=next.getWorldPosition(new THREE.Vector3()).sub(proximal.getWorldPosition(new THREE.Vector3()));
    const towardPalm=middle.getWorldPosition(new THREE.Vector3()).sub(proximal.getWorldPosition(new THREE.Vector3()));
    const axis=new THREE.Vector3().crossVectors(direction,towardPalm);
    if(axis.lengthSq()>1e-10){
      axis.normalize().applyQuaternion(bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert());
      const angle=thumbCurl(transform(b.clone().sub(a)),transform(c.clone().sub(b)))*(segment==='3'?.45:1);
      bone.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(axis,angle));
    }
  }
  bone.updateMatrixWorld(true);return true;
}
function fitFinger(bone,name,rig,positions,transform){
  const match=name.match(/^(index|middle|ring|pinky)0([123])([lr])$/);
  if(!match)return false;
  const [,digit,segment,side]=match,hand=side==='l'?'lefthand':'righthand';
  const wrist=positions.get(hand),middle=positions.get(`middle01${side}`),index=positions.get(`index01${side}`),little=positions.get(`pinky01${side}`);
  if(!wrist||!middle||!index||!little)return false;
  const axis=transform(index.clone().sub(little)).normalize();
  // Source phalange angles transfer curl, while the avatar's bind pose keeps
  // its own finger spacing. A distal joint without a tip follows the PIP curl.
  const k=Math.min(Number(segment),2),base=positions.get(`${digit}0${k}${side}`),next=positions.get(`${digit}0${k+1}${side}`);
  const previous=k===1?wrist:positions.get(`${digit}01${side}`);
  if(!base||!next||!previous)return false;
  const incoming=k===1?middle.clone().sub(wrist):base.clone().sub(previous);
  const angle=fingerBend(transform(incoming),transform(next.clone().sub(base)),axis,k===1?.85:1.45)*(segment==='3'?.65:1);
  const handBone=rig.bones.get(hand),targetMiddle=rig.bones.get(`middle01${side}`),targetIndex=rig.bones.get(`index01${side}`),targetLittle=rig.bones.get(`pinky01${side}`);
  if(!handBone||!targetMiddle||!targetIndex||!targetLittle)return false;
  const targetFrame=palmFrame(targetMiddle.getWorldPosition(new THREE.Vector3()).sub(handBone.getWorldPosition(new THREE.Vector3())),targetIndex.getWorldPosition(new THREE.Vector3()).sub(targetLittle.getWorldPosition(new THREE.Vector3())));
  if(!targetFrame)return false;
  const hinge=new THREE.Vector3(1,0,0).applyQuaternion(targetFrame).applyQuaternion(bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert());
  bone.quaternion.copy(rig.bind.get(bone)).premultiply(new THREE.Quaternion().setFromAxisAngle(hinge,angle));
  bone.updateMatrixWorld(true);return true;
}
function fitPalm(bone,bones,positions,transform){
  const side=key(bone.name)==='lefthand'?'l':'r';
  const middle=`middle01${side}`,index=`index01${side}`,little=`pinky01${side}`;
  if(![middle,index,little,key(bone.name)].every(k=>positions.has(k))||![middle,index,little].every(k=>bones.has(k)))return false;
  const origin=bone.getWorldPosition(new THREE.Vector3());
  const current=palmFrame(bones.get(middle).getWorldPosition(new THREE.Vector3()).sub(origin),bones.get(index).getWorldPosition(new THREE.Vector3()).sub(bones.get(little).getWorldPosition(new THREE.Vector3())));
  const sourceOrigin=positions.get(key(bone.name));
  const desired=palmFrame(transform(positions.get(middle).clone().sub(sourceOrigin)),transform(positions.get(index).clone().sub(positions.get(little))));
  if(!current||!desired)return false;
  const delta=desired.multiply(current.invert()),parent=bone.parent.getWorldQuaternion(new THREE.Quaternion());
  bone.quaternion.premultiply(parent.clone().invert().multiply(delta).multiply(parent));bone.updateMatrixWorld(true);return true;
}
export function bindRelativeWorldRotation(sourceWorld,sourceRestWorld,targetRestWorld,parentWorld,basis=new THREE.Quaternion()){
  const delta=sourceWorld.clone().multiply(sourceRestWorld.clone().invert());
  const worldDelta=basis.clone().multiply(delta).multiply(basis.clone().invert());
  const desired=worldDelta.multiply(targetRestWorld);
  return parentWorld.clone().invert().multiply(desired);
}

// ---------------------------------------------------------------------------
// Facial expressions: seven basic emotions, each with three authored ARKit
// presets. Weak, medium and strong use different action-unit combinations
// (FACS-inspired), not one shape scaled linearly. Bilateral names expand to
// their Left/Right channels; explicit Left/Right names stay asymmetric.
const BILATERAL=new Set(['browDown','browOuterUp','cheekSquint','eyeBlink','eyeLookDown','eyeLookUp','eyeSquint','eyeWide','mouthDimple','mouthFrown','mouthLowerDown','mouthPress','mouthSmile','mouthStretch','mouthUpperUp','noseSneer']);
function expandPreset(preset){
  const out={};
  for(const [name,value] of Object.entries(preset))if(BILATERAL.has(name)){out[name+'Left']=Math.max(out[name+'Left']||0,value);out[name+'Right']=Math.max(out[name+'Right']||0,value);}
  for(const [name,value] of Object.entries(preset))if(!BILATERAL.has(name))out[name]=value;
  return out;
}
const RAW_PRESETS={
  neutral:[
    {},
    {mouthClose:.04,eyeSquint:.04},
    {mouthPress:.14,browDown:.08,eyeSquint:.1,mouthRollLower:.05}
  ],
  happiness:[
    {mouthSmile:.32,cheekSquint:.14,mouthDimple:.1},
    {mouthSmile:.58,cheekSquint:.4,eyeSquint:.2,mouthDimple:.22,mouthUpperUp:.08},
    {mouthSmile:.88,cheekSquint:.72,eyeSquint:.42,jawOpen:.2,mouthUpperUp:.3,mouthLowerDown:.26,mouthDimple:.3,browOuterUp:.12}
  ],
  sadness:[
    {browInnerUp:.36,mouthFrown:.2},
    {browInnerUp:.62,browDown:.14,mouthFrown:.45,mouthShrugLower:.22,eyeSquint:.1,eyeLookDown:.15},
    {browInnerUp:.88,browDown:.3,mouthFrown:.72,mouthShrugLower:.45,mouthPress:.22,mouthStretch:.14,eyeSquint:.24,eyeBlink:.22,eyeLookDown:.3}
  ],
  anger:[
    {browDown:.36,eyeSquint:.16,mouthPress:.16},
    {browDown:.66,eyeSquint:.3,eyeWide:.12,mouthPress:.36,noseSneer:.22,mouthRollLower:.12},
    {browDown:.95,noseSneer:.46,eyeWide:.26,eyeSquint:.34,mouthUpperUp:.32,mouthStretch:.3,mouthFrown:.3,mouthLowerDown:.22,jawForward:.15}
  ],
  disgust:[
    {noseSneer:.32,mouthUpperUp:.16,browDown:.1},
    {noseSneer:.62,mouthUpperUp:.36,cheekSquint:.25,browDown:.3,mouthFrown:.2,mouthShrugUpper:.22},
    {noseSneer:.9,mouthUpperUpLeft:.7,mouthUpperUpRight:.5,cheekSquint:.5,eyeSquint:.35,browDown:.46,mouthFrown:.4,mouthShrugLower:.35,mouthLeft:.12,jawOpen:.05}
  ],
  fear:[
    {browInnerUp:.42,eyeWide:.26,mouthStretch:.1},
    {browInnerUp:.7,browOuterUp:.3,browDown:.15,eyeWide:.52,mouthStretch:.36,jawOpen:.1},
    {browInnerUp:.95,browOuterUp:.52,browDown:.26,eyeWide:.86,mouthStretch:.66,jawOpen:.3,mouthLowerDown:.32,mouthFrown:.18}
  ],
  surprise:[
    {browInnerUp:.36,browOuterUp:.32,eyeWide:.22},
    {browInnerUp:.66,browOuterUp:.62,eyeWide:.52,jawOpen:.2,mouthFunnel:.1},
    {browInnerUp:.95,browOuterUp:.92,eyeWide:.86,jawOpen:.5,mouthFunnel:.22,mouthLowerDown:.12}
  ]
};
export const EXPRESSION_PRESETS=Object.freeze(Object.fromEntries(Object.entries(RAW_PRESETS).map(([name,levels])=>[name,Object.freeze(levels.map(level=>Object.freeze(expandPreset(level))))])));
export const EMOTIONS=Object.freeze(Object.keys(EXPRESSION_PRESETS));
export const EMOTION_ALIASES=Object.freeze({joy:'happiness',happy:'happiness',joyful:'happiness',angry:'anger',mad:'anger',sad:'sadness',scared:'fear',afraid:'fear',fearful:'fear',disgusted:'disgust',surprised:'surprise',calm:'neutral',none:'neutral'});
// Intensity anchors for neutral(0), weak, medium and strong.
export const EXPRESSION_LEVEL_INTENSITY=Object.freeze([0,.3,.6,1]);
export function normalizeEmotion(name){
  const n=String(name??'neutral').toLowerCase().trim();
  return EXPRESSION_PRESETS[n]?n:EMOTION_ALIASES[n]||'neutral';
}
const LEVEL_WORDS={weak:1,low:1,mild:1,subtle:1,slight:1,medium:2,moderate:2,mid:2,strong:3,high:3,intense:3,extreme:3};
// Integers 1, 2, 3 (or weak/medium/strong) select a preset; fractional
// numbers in [0,1] are a continuous intensity blended between presets.
export function expressionIntensity(level){
  if(typeof level==='string'){const word=LEVEL_WORDS[level.toLowerCase().trim()];if(word)return EXPRESSION_LEVEL_INTENSITY[word];level=Number(level);}
  if(level&&typeof level==='object'&&'level' in level)return expressionIntensity(Math.round(Number(level.level)||0)||0);
  const n=Number(level);
  if(!Number.isFinite(n))return EXPRESSION_LEVEL_INTENSITY[2];
  if(Number.isInteger(n)&&n>=1&&n<=3)return EXPRESSION_LEVEL_INTENSITY[n];
  return clamp(n);
}
export function expressionChannels(name='neutral',intensity=.6){
  const presets=EXPRESSION_PRESETS[normalizeEmotion(name)],i=clamp(intensity),anchors=EXPRESSION_LEVEL_INTENSITY;
  const stops=[{},...presets];
  let k=0;while(k<anchors.length-2&&i>anchors[k+1])k++;
  const t=(i-anchors[k])/(anchors[k+1]-anchors[k]),a=stops[k],b=stops[k+1],out={};
  for(const channel of new Set([...Object.keys(a),...Object.keys(b)])){
    const value=(a[channel]||0)*(1-t)+(b[channel]||0)*t;
    if(value>1e-4)out[channel]=value;
  }
  return out;
}

const MOUTH_SHAPE=/^(jawOpen|jawForward|mouthFunnel|mouthPucker|mouthLowerDown|mouthStretch|mouthClose|mouthRoll|mouthPress|mouthShrug)/;
// Compose recorded face data, emotion, blink and speech mouth channels.
export function composeFaceChannels(actor){
  const f={...(actor.face||{})};
  const lipSync=Boolean(actor.speaking&&actor.visemes);
  const expression=actor.expressionCurrent||expressionChannels(actor.emotion,actor.intensity);
  for(const [channel,value] of Object.entries(expression)){
    // Visemes own the jaw and lip aperture while speaking; keep the emotional
    // brows, eyes, cheeks and smile, and only a trace of the mouth shape.
    const v=lipSync&&MOUTH_SHAPE.test(channel)?value*.35:value;
    f[channel]=Math.max(f[channel]||0,v);
  }
  const blink=clamp(actor.blinkValue);
  if(blink>0)for(const side of ['Left','Right']){
    f['eyeBlink'+side]=Math.max(f['eyeBlink'+side]||0,blink);
    if(f['eyeWide'+side])f['eyeWide'+side]*=1-blink;
  }
  if(lipSync){
    for(const [channel,value] of Object.entries(actor.visemes))f[channel]=Math.max(f[channel]||0,clamp(value));
  }else if(actor.speaking){
    // This is an approximate animated speaking envelope, not phoneme alignment.
    const pulse=clamp(actor.speechLevel ?? .08);
    f.jawOpen=Math.max(f.jawOpen||0,pulse*.55);
    f.viseme_aa=Math.max(f.viseme_aa||0,pulse*.3);
  }
  return f;
}

// ---------------------------------------------------------------------------
// Small rig helpers shared by IK, gaze, idle and posture overlays.
const worldPosition=object=>object.getWorldPosition(new THREE.Vector3());
export function rotateBoneWorld(bone,axis,angle){
  if(!bone?.parent||!angle||axis.lengthSq()<1e-10)return false;
  const parentWorld=bone.parent.getWorldQuaternion(new THREE.Quaternion());
  const delta=new THREE.Quaternion().setFromAxisAngle(axis.clone().normalize(),angle);
  bone.quaternion.premultiply(parentWorld.clone().invert().multiply(delta).multiply(parentWorld));
  bone.updateMatrixWorld(true);return true;
}
// Analytic two-bone IK in world space. The middle joint bends in the plane
// containing the pole hint; unreachable targets extend the limb toward them.
export function solveTwoBoneIK(upper,lower,end,target,pole=null,weight=1){
  weight=clamp(weight);
  if(!upper?.parent||!lower||!end||!target||weight<=0)return false;
  const a=worldPosition(upper),b=worldPosition(lower),c=worldPosition(end);
  const l1=a.distanceTo(b),l2=b.distanceTo(c),toTarget=target.clone().sub(a);
  let d=toTarget.length();
  if(l1<1e-6||l2<1e-6||d<1e-6)return false;
  const dir=toTarget.divideScalar(d);
  d=Math.max(Math.abs(l1-l2)+1e-4,Math.min(l1+l2-1e-4,d));
  const orthogonal=v=>v.addScaledVector(dir,-v.dot(dir));
  let bend=orthogonal(pole?pole.clone().sub(a):b.clone().sub(a));
  if(bend.lengthSq()<1e-8)bend=orthogonal(b.clone().sub(a));
  if(bend.lengthSq()<1e-8)bend=orthogonal(Math.abs(dir.y)<.9?new THREE.Vector3(0,-1,0):new THREE.Vector3(0,0,1));
  bend.normalize();
  const cosine=Math.max(-1,Math.min(1,(l1*l1+d*d-l2*l2)/(2*l1*d))),sine=Math.sqrt(1-cosine*cosine);
  const joint=a.clone().addScaledVector(dir,l1*cosine).addScaledVector(bend,l1*sine);
  const reach=a.clone().addScaledVector(dir,d);
  const upperStart=upper.quaternion.clone(),lowerStart=lower.quaternion.clone();
  aimBoneToward(upper,lower,joint.clone().sub(a));
  aimBoneToward(lower,end,reach.sub(worldPosition(lower)));
  if(weight<1){
    const upperSolved=upper.quaternion.clone(),lowerSolved=lower.quaternion.clone();
    upper.quaternion.copy(upperStart).slerp(upperSolved,weight);lower.quaternion.copy(lowerStart).slerp(lowerSolved,weight);
    upper.updateMatrixWorld(true);
  }
  return true;
}
function resolvePoint(target,camera=null){
  if(target==null)return null;
  if(target==='camera')return camera?worldPosition(camera):null;
  if(target.isObject3D)return worldPosition(target);
  if(target.isVector3)return target.clone();
  if(Array.isArray(target)&&target.length>=3&&target.slice(0,3).every(Number.isFinite))return new THREE.Vector3(target[0],target[1],target[2]);
  if(typeof target==='object'&&['x','y','z'].every(k=>Number.isFinite(target[k])))return new THREE.Vector3(target.x,target.y,target.z);
  return null;
}
// Depth-only silhouette geometry hides avatar parts behind real objects.
export function makeOccluder(object,{renderOrder=-10}={}){
  object?.traverse?.(o=>{
    if(!o.isMesh)return;
    const source=Array.isArray(o.material)?o.material:[o.material];
    const materials=source.map(m=>new THREE.MeshBasicMaterial({colorWrite:false,depthWrite:true,depthTest:true,side:m?.side??THREE.FrontSide}));
    if(!o.userData.occluder)o.userData.originalMaterial=o.material;
    o.material=Array.isArray(o.material)?materials:materials[0];
    o.renderOrder=renderOrder;o.userData.occluder=true;
  });
  return object;
}
function measureModel(model){
  // Measured before the model joins the actor root: coordinates are actor-local.
  model.updateMatrixWorld(true);
  const box=new THREE.Box3().setFromObject(model),bones=new Map();
  model.traverse(o=>{if(o.isBone&&!bones.has(key(o.name)))bones.set(key(o.name),o);});
  const y=name=>bones.has(name)?worldPosition(bones.get(name)).y:null;
  const forward=new THREE.Vector3(0,0,1),forwardLocal=new Map();
  for(const name of ['neck','head']){const bone=bones.get(name);if(bone)forwardLocal.set(bone,forward.clone().applyQuaternion(bone.getWorldQuaternion(new THREE.Quaternion()).invert()));}
  const height=Math.max(.5,box.max.y-Math.min(0,box.min.y));
  const feet=[y('leftfoot'),y('rightfoot')].filter(v=>v!==null);
  return {box,height,hipsY:y('hips')??height*.53,ankleY:feet.length?feet.reduce((a,b)=>a+b,0)/feet.length:height*.05,forwardLocal,
    base:{position:model.position.clone(),quaternion:model.quaternion.clone()}};
}
function createRig(model,metrics){
  const rig={model,bones:new Map(),bind:new Map(),restWorld:new Map(),morphs:[],morphDefaults:new Map(),morphKeys:new Map(),metrics};
  model.traverse(o=>{
    if(o.isBone){const name=key(o.name);if(!rig.bones.has(name))rig.bones.set(name,o);rig.bind.set(o,o.quaternion.clone());}
    if(o.isMesh){o.frustumCulled=false;if(o.morphTargetDictionary&&o.morphTargetInfluences){rig.morphs.push(o);rig.morphDefaults.set(o,o.morphTargetInfluences.slice());
      rig.morphKeys.set(o,Object.entries(o.morphTargetDictionary).map(([name,index])=>[morphName(name),index]).filter(([name])=>/^(brow|cheek|eye|jaw|mouth|nose|tongue|viseme)/.test(name)));}}
  });
  for(const bone of rig.bind.keys())rig.restWorld.set(bone,bone.getWorldQuaternion(new THREE.Quaternion()));
  return rig;
}

// Palm normal of a skinned hand from its finger bases: fingers x (index - little), signed per side so it
// points out of the palm (T-pose palms face down for both hands).
export function palmNormal(rig,side){
  const s=side==='left'?'l':'r',hand=rig.bones.get(side+'hand'),middle=rig.bones.get(`middle01${s}`),index=rig.bones.get(`index01${s}`),little=rig.bones.get(`pinky01${s}`);
  if(!hand||!middle||!index||!little)return null;
  const at=b=>b.getWorldPosition(new THREE.Vector3());
  const normal=new THREE.Vector3().crossVectors(at(middle).sub(at(hand)),at(index).sub(at(little))).multiplyScalar(side==='left'?1:-1);
  return normal.lengthSq()>1e-12?normal.normalize():null;
}
// Pronation/supination: twist the forearm (most) and the wrist (rest) about the forearm axis so the palm
// normal turns toward ``target`` (world space). Rigs without finger bones are left unchanged.
function turnPalm(rig,side,target){
  const fore=rig.bones.get(side+'forearm'),hand=rig.bones.get(side+'hand');
  if(!fore||!hand||!fore.parent||!hand.parent)return false;
  const twist=(bone,share)=>{
    const normal=palmNormal(rig,side);if(!normal)return false;
    const axis=hand.getWorldPosition(new THREE.Vector3()).sub(fore.getWorldPosition(new THREE.Vector3()));
    if(axis.lengthSq()<1e-10)return false;axis.normalize();
    const p=normal.clone().addScaledVector(axis,-normal.dot(axis)),q=target.clone().addScaledVector(axis,-target.dot(axis));
    if(p.lengthSq()<1e-10||q.lengthSq()<1e-10)return false;
    const angle=Math.atan2(axis.dot(new THREE.Vector3().crossVectors(p,q)),p.dot(q))*share;
    const parent=bone.parent.getWorldQuaternion(new THREE.Quaternion());
    bone.quaternion.premultiply(parent.clone().invert().multiply(new THREE.Quaternion().setFromAxisAngle(axis,angle)).multiply(parent));
    bone.updateMatrixWorld(true);return true;
  };
  return twist(fore,.6)&&twist(hand,1);
}

// World-space anatomical directions avoid assuming that imported skinning
// bones share the local Euler axes of the old procedural stick character.
export function applyGesturePose(rig,name='idle',t=0){
  const n=String(name).toLowerCase(),beat=Math.sin(t*4),open=/welcome|open|explain|offer|reassure|present/.test(n),point=/point|touch|reach|turn|switch|warning/.test(n),wave=/wave|hello|greet/.test(n),think=/think|consider/.test(n),sit=/sit/.test(n),walk=/walk|move/.test(n);
  const basis=rig.model.getWorldQuaternion(new THREE.Quaternion());
  for(const [bone,bind] of rig.bind)bone.quaternion.copy(bind);
  rig.model.updateMatrixWorld(true);
  const aim=(parent,child,direction)=>{const bone=rig.bones.get(parent),end=rig.bones.get(child);if(bone&&end)aimBoneToward(bone,end,new THREE.Vector3(...direction).applyQuaternion(basis));};
  for(const side of ['left','right']){
    const sign=side==='left'?1:-1;
    let upper=[sign*.18,-1,.05],lower=[sign*.06,-1,.12];
    // Open-palm presentation: the upper arm hangs close to the torso with the elbow slightly forward, the
    // forearm comes forward (a little out, a little down) so the wrists sit at waist-to-lower-chest height
    // about shoulder width apart, palms turned up/inward. Wider or higher targets read as a T-pose.
    if(open){upper=[sign*.08,-.95,.3];lower=[sign*(.24+.06*beat),-.14+.04*beat,.96];}
    if((point||wave||think)&&side==='right'){
      upper=think?[-.25,-.65,.65]:wave?[-.75,.35,.35]:[-.65,-.12,.8];
      lower=think?[.12,.9,.2]:wave?[-.15,1,.15+.3*beat]:[-.7,.05+.08*beat,.8];
    }
    if(/beat/.test(n)&&side==='right'){upper=[-.3,-.8,.35];lower=[-.3,.15+.3*beat,.85];}
    if(walk){upper=[sign*.12,-1,(side==='left'?1:-1)*.3*Math.sin(t*7)];lower=[sign*.06,-1,.1];}
    aim(side+'arm',side+'forearm',upper);aim(side+'forearm',side+'hand',lower);
    if(open)turnPalm(rig,side,new THREE.Vector3(-sign*.25,.93,.25).applyQuaternion(basis));
    if(sit){aim(side+'upleg',side+'leg',[sign*.12,-.15,.9]);aim(side+'leg',side+'foot',[0,-1,.1]);}
    else if(walk){aim(side+'upleg',side+'leg',[sign*.06,-1,(side==='left'?1:-1)*.3*Math.sin(t*7)]);}
  }
  if(/nod|yes|agree/.test(n)){const head=rig.bones.get('head');if(head)head.quaternion.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(.1*Math.sin(t*4),0,0)));}
}

const AUTO_SIT=/(^|[^a-z])(sit|sits|sitting|seated|sit_down)([^a-z]|$)/,AUTO_LIE=/(^|[^a-z])(lie|lies|lying|lie_down|sleep|sleeping)([^a-z]|$)/;
const smoothstep=x=>{x=Math.max(0,Math.min(1,x));return x*x*(3-2*x);};
export function blinkCurve(elapsed){
  if(!(elapsed>=0))return 0;
  if(elapsed<.07)return smoothstep(elapsed/.07);
  if(elapsed<.1)return 1;
  return elapsed<.22?1-smoothstep((elapsed-.1)/.12):0;
}

export function createStage(container, options={}) {
  const scene=new THREE.Scene();scene.background=new THREE.Color(options.background||'#101827');
  const camera=new THREE.PerspectiveCamera(42,1,.01,100);camera.position.set(0,1.6,5.8);camera.lookAt(0,1.15,0);
  const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  // CSS size follows the container; drawing-buffer size includes pixel ratio.
  // Intrinsic canvas sizing otherwise feeds back into grid layout on HiDPI.
  Object.assign(renderer.domElement.style,{display:'block',width:'100%',height:'100%'});container.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xe6f4ff,0x344055,2.4));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(3,5,4);scene.add(light);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.MeshStandardMaterial({color:0x182438,roughness:.9}));floor.rotation.x=-Math.PI/2;scene.add(floor);
  const grid=new THREE.GridHelper(12,24,0x405575,0x24344b);grid.position.y=.002;scene.add(grid);
  const actors=[],props=new Map(),clock=new THREE.Clock(),loader=new GLTFLoader();let raf,disposed=false,lastTime=0;
  const modelChoices={rowan:new URL('./avatars/rowan.glb',import.meta.url).href,mira:new URL('./avatars/mira.glb',import.meta.url).href};
  const defaultIdle=()=>options.idle===false?{blink:false,breath:false,head:false}:{blink:true,breath:true,head:true,...(typeof options.idle==='object'?options.idle:{})};
  function attachModel(actor,model){
    const metrics=measureModel(model);
    actor.root.add(model);model.updateMatrixWorld(true);
    return createRig(model,metrics);
  }
  function makeAvatar(color=0x60c8d9,x=0,z=0,avatarUrl=options.avatarUrl===undefined?new URL('./avatars/rowan.glb',import.meta.url).href:options.avatarUrl) {
    const root=new THREE.Group();root.position.set(x,0,z);scene.add(root);
    const procedural=new THREE.Group();root.add(procedural);
    const mat=new THREE.MeshStandardMaterial({color,roughness:.65}),skin=new THREE.MeshStandardMaterial({color:0xe1ad91,roughness:.8}),dark=new THREE.MeshStandardMaterial({color:0x182030});
    function mesh(geometry,material,parent=procedural,pos=[0,0,0]){const m=new THREE.Mesh(geometry,material);m.position.set(...pos);parent.add(m);return m;}
    mesh(new THREE.CapsuleGeometry(.25,.5,5,12),mat,procedural,[0,1.15,0]);
    const head=new THREE.Group();head.position.y=1.88;procedural.add(head);mesh(new THREE.SphereGeometry(.24,24,20),skin,head);
    const eyes=[-.085,.085].map(v=>mesh(new THREE.SphereGeometry(.034,12,8),dark,head,[v,.025,.218]));
    const brows=[-.085,.085].map(v=>mesh(new THREE.BoxGeometry(.105,.014,.015),dark,head,[v,.092,.217]));
    const mouth=mesh(new THREE.SphereGeometry(.06,16,8),dark,head,[0,-.085,.225]);mouth.scale.set(1,.15,.25);
    function limb(x,y,length){const pivot=new THREE.Group();pivot.position.set(x,y,0);procedural.add(pivot);mesh(new THREE.CapsuleGeometry(.065,length,4,8),mat,pivot,[0,-length/2,0]);return pivot;}
    const arms=[limb(-.32,1.52,.58),limb(.32,1.52,.58)],legs=[limb(-.13,.76,.65),limb(.13,.76,.65)];
    const actor={root,procedural,head,arms,legs,mouth,eyes,brows,gesture:'idle',emotion:'neutral',intensity:.5,speaking:false,walking:false,face:null,move:null,rig:null,loadError:null,
      idle:defaultIdle(),phase:Math.random()*20,blinkStart:-1,blinkNext:null,blinkValue:0,visemes:null,reach:{},gaze:null,posture:undefined,overlayBase:null,expressionCurrent:null};
    actors.push(actor);
    const request=actor.loadRequest=1;
    actor.ready=avatarUrl?new Promise(resolve=>loader.load(avatarUrl,gltf=>{
      if(disposed){disposeObject(gltf.scene);resolve({loaded:false,error:new Error('Stage disposed')});return;}
      if(request!==actor.loadRequest){disposeObject(gltf.scene);resolve({loaded:false,stale:true,actor});return;}
      const rig=attachModel(actor,gltf.scene);
      if(actor.rig){root.remove(actor.rig.model);disposeObject(actor.rig.model);}
      actor.rig=rig;actor.overlayBase=null;actor.loadError=null;procedural.visible=false;resolve({loaded:true,actor});
    },undefined,error=>{if(request===actor.loadRequest){actor.loadError=error;console.error(`Avatar could not load: ${avatarUrl}`,error);options.onAvatarError?.(error,actor);}resolve({loaded:false,error,actor});})):Promise.resolve({loaded:false,actor});
    return actor;
  }
  const avatar=makeAvatar(options.color||0x60c8d9);const ready=avatar.ready.then(result=>({...result,actors}));
  const status=document.createElement('div');status.className='avatar-load-status';status.style.cssText='position:absolute;left:8px;bottom:8px;z-index:2;padding:3px 7px;border-radius:4px;background:#182438dd;color:#fff;font:11px system-ui;pointer-events:none';
  const previousPosition=container.style.position;if(!previousPosition)container.style.position='relative';container.append(status);
  const selector=document.createElement('select');selector.setAttribute('aria-label','Avatar character');selector.style.cssText='position:absolute;right:8px;top:8px;z-index:2;width:auto;max-width:130px;min-width:90px;padding:4px;border-radius:4px;background:#182438;color:#fff;border:1px solid #7e99ad';
  for(const label of Object.keys(modelChoices)){const option=document.createElement('option');option.value=label;option.textContent=label[0].toUpperCase()+label.slice(1);selector.append(option);}container.append(selector);
  function loadCharacter(actor,url){
    const request=actor.loadRequest=(actor.loadRequest||0)+1;
    actor.ready=new Promise(resolve=>loader.load(url,gltf=>{
      if(disposed){disposeObject(gltf.scene);resolve({loaded:false,error:new Error('Stage disposed'),actor});return;}
      if(request!==actor.loadRequest){disposeObject(gltf.scene);resolve({loaded:false,stale:true,actor});return;}
      const rig=attachModel(actor,gltf.scene);
      if(actor.rig){actor.root.remove(actor.rig.model);disposeObject(actor.rig.model);}actor.rig=rig;actor.overlayBase=null;actor.procedural.visible=false;actor.loadError=null;resolve({loaded:true,actor});
    },undefined,error=>{if(request===actor.loadRequest){actor.loadError=error;console.error(`Avatar could not load: ${url}`,error);options.onAvatarError?.(error,actor);}resolve({loaded:false,error,actor});}));
    return actor.ready;
  }
  function setCharacter(name,actor=avatar){const url=modelChoices[name]||name;status.textContent=`Loading ${name}…`;return loadCharacter(actor,url).then(result=>{if(!result.stale)status.textContent=result.loaded?`${name} ready`:`Avatar failed to load: ${name}`;return result;});}
  selector.onchange=()=>{Promise.all(actors.map(actor=>setCharacter(selector.value,actor)));};
  status.textContent='Loading avatar…';ready.then(result=>{if(!result.stale)status.textContent=result.loaded?'Avatar ready':'Avatar unavailable; procedural fallback active';});
  const skeleton=new THREE.Group();scene.add(skeleton);skeleton.visible=false;let points=[],bones=[];
  function setSkeleton(joints,edges=[]){
    if(!joints?.length)return;
    skeleton.visible=true;const vectors=joints.map(p=>new THREE.Vector3(Number(p[0])||0,Number(p[1])||0,Number(p[2])||0));
    if(points.length!==vectors.length){for(const o of [...points,...bones]){skeleton.remove(o);o.geometry.dispose();o.material.dispose();}points=[];bones=[];for(const v of vectors){const o=new THREE.Mesh(new THREE.SphereGeometry(.035,8,6),new THREE.MeshBasicMaterial({color:0xa9e879}));skeleton.add(o);points.push(o);}}
    points.forEach((o,i)=>o.position.copy(vectors[i]));
    while(bones.length<edges.length){const o=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x81cee5}));skeleton.add(o);bones.push(o);}
    bones.forEach((o,i)=>{o.visible=i<edges.length;if(o.visible)o.geometry.setFromPoints([vectors[edges[i][0]],vectors[edges[i][1]]]);});
    avatar.root.visible=false;
  }
  // Overlays (idle, IK, gaze, seated legs) are layered on the current base pose
  // each frame. The base is restored before the next frame or the next clip
  // frame, so overlays never accumulate on recorded motion.
  function restoreOverlay(actor){
    if(!actor.overlayBase)return;
    for(const [bone,value] of actor.overlayBase)bone.quaternion.copy(value);
    actor.overlayBase=null;actor.rig?.model.updateMatrixWorld(true);
  }
  function overlay(actor,...names){
    const list=[];
    for(const name of names){const bone=actor.rig.bones.get(name);if(!bone)continue;if(!actor.overlayBase.has(bone))actor.overlayBase.set(bone,bone.quaternion.clone());list.push(bone);}
    return list;
  }
  function setMotionFrame(frame,actor=avatar,source={}){
    if(!actor.rig||!frame)return false;
    restoreOverlay(actor);
    const rotations=frame.quaternions||frame.rotations||frame.rotation6d||[];
    const names=frame.jointNames||frame.joint_order||source.jointNames||source.joint_order||[];
    const type=frame.rotation6d?'rotation6d':'quaternion';
    const basis=quaternion(source.restBasis)||new THREE.Quaternion();
    const signs=source.axisSigns||[1,1,1];
    const parents=frame.parents||source.parents||[];
    const local=rotations.map(value=>quaternion(value,type)||new THREE.Quaternion());
    const rest=local.map((_,i)=>quaternion(source.restQuaternions?.[i])||new THREE.Quaternion());
    const world=(values,i,cache,active=new Set())=>{
      if(cache[i])return cache[i];
      if(active.has(i))return values[i].clone();
      active.add(i);const p=parents[i];
      const result=Number.isInteger(p)&&p>=0&&p<values.length&&p!==i?world(values,p,cache,active).clone().multiply(values[i]):values[i].clone();
      active.delete(i);cache[i]=result;return result;
    };
    const frameWorld=[],restWorld=[];
    for(const [bone,bind] of actor.rig.bind)bone.quaternion.copy(bind);
    actor.rig.model.updateMatrixWorld(true);
    const indices=local.map((_,i)=>i).sort((a,b)=>{
      const depth=i=>{let count=0,seen=new Set();for(let p=parents[i];Number.isInteger(p)&&p>=0&&p<parents.length&&!seen.has(p);p=parents[p]){seen.add(p);count++;}return count;};
      return depth(a)-depth(b);
    });
    let applied=0;
    for(const index of indices){
      const bone=actor.rig.bones.get(key(names[index]));if(!bone)continue;
      const parentWorld=bone.parent.getWorldQuaternion(new THREE.Quaternion());
      bone.quaternion.copy(bindRelativeWorldRotation(reflectQuaternion(world(local,index,frameWorld),signs),reflectQuaternion(world(rest,index,restWorld),signs),actor.rig.restWorld.get(bone),parentWorld,basis));
      bone.updateMatrixWorld(true);applied++;
    }
    // Canonical humanoid rotation axes are not MPFB skinning bind axes.
    // Applying their twist before fitting landmarks collapses the arm meshes.
    // Fit FK landmarks from the target bind pose; never retain uncalibrated
    // source twist. Quaternion-only inputs keep the explicit delta path above.
    if(Array.isArray(frame.positions)&&setPosePositions(frame.positions,names,actor,{basis:source.restBasis,axisSigns:signs}))applied=Math.max(1,applied);
    actor.motionActive=applied>0;
    // An explicit seat or bed anchors the root; clip root travel would slide the actor off it.
    const anchored=['sit','lie'].includes(actor.posture?.type);
    if(frame.rootTranslation&&!anchored){const p=new THREE.Vector3(...frame.rootTranslation.map((v,i)=>(Number(v)||0)*signs[i])).applyQuaternion(basis);actor.root.position.copy(p);}
    return applied>0;
  }
  function setPosePositions(joints,jointNames,actor=avatar,options={}){
    if(!actor.rig||!Array.isArray(joints)||!Array.isArray(jointNames))return false;
    restoreOverlay(actor);
    const positions=new Map();
    for(let i=0;i<jointNames.length;i++){
      const p=joints[i];if(p?.length>=3&&p.every(Number.isFinite))positions.set(key(jointNames[i]),new THREE.Vector3(...p));
    }
    if(positions.size<2)return false;
    const basis=options.basis?.length===4?quaternion(options.basis):null;
    const transform=p=>{const v=p.clone();if(options.axisSigns)v.multiply(new THREE.Vector3(...options.axisSigns));return basis?v.applyQuaternion(basis):v;};
    // Restore bind rotations, then solve parent-to-child directions in world space.
    if(!options.preservePose)for(const [bone,bind] of actor.rig.bind)bone.quaternion.copy(bind);
    actor.rig.model.updateMatrixWorld(true);
    const entries=[...actor.rig.bones.entries()].sort((a,b)=>{
      const depth=o=>{let n=0;for(let p=o.parent;p&&p!==actor.rig.model;p=p.parent)n++;return n;};
      return depth(a[1])-depth(b[1]);
    });
    let applied=0;
    for(const [name,bone] of entries){
      // Sparse position clips locate the head but do not determine its facing
      // direction. Keep the neck/head bind orientation instead of fitting it
      // from a single landmark; full quaternion clips retain head motion.
      if(name==='neck'||name==='head')continue;
      if((name==='lefthand'||name==='righthand')&&fitPalm(bone,actor.rig.bones,positions,transform)){applied++;continue;}
      if(fitThumb(bone,name,actor.rig,positions,transform)){applied++;continue;}
      if(fitFinger(bone,name,actor.rig,positions,transform)){applied++;continue;}
      const from=positions.get(name);if(!from)continue;
      const child=poseDirectionChild(bone,positions);
      if(!child)continue;
      const target=transform(positions.get(key(child.name)).clone().sub(from));
      if(aimBoneToward(bone,child,target)){
        const finger=name.match(/^(thumb|index|middle|ring|pinky)0([123])[lr]$/);
        if(finger){
          // Different finger proportions and noisy capture can produce extreme
          // local rotations. Retain articulated motion within a bounded range.
          const limit=(finger[1]==='thumb'?[1.1,1.4,1.0]:[1.4,1.9,1.4])[Number(finger[2])-1];
          const bind=actor.rig.bind.get(bone),angle=bind.angleTo(bone.quaternion);
          if(angle>limit)bone.quaternion.copy(bind.clone().slerp(bone.quaternion,limit/angle));
          bone.updateMatrixWorld(true);
        }
        applied++;
      }
    }
    actor.motionActive=applied>0;
    return applied>0;
  }
  function clearMotion(actor=avatar){actor.motionActive=false;actor.overlayBase=null;if(actor.rig)for(const [bone,bind] of actor.rig.bind)bone.quaternion.copy(bind);}
  function expression(name='neutral',intensity=.5,actor=avatar){actor.emotion=name;actor.intensity=clamp(intensity);actor.expressionLevel=null;}
  // setExpression('anger',2) selects the medium preset; .45 blends weak→medium.
  function setExpression(name='neutral',level=2,actor=avatar){
    actor.emotion=normalizeEmotion(name);actor.intensity=expressionIntensity(level);
    const index=EXPRESSION_LEVEL_INTENSITY.indexOf(actor.intensity);actor.expressionLevel=index>0?index:null;
    return {emotion:actor.emotion,intensity:actor.intensity,level:actor.expressionLevel};
  }
  function gesture(name='idle',actor=avatar,time=null){actor.gesture=name;actor.gestureTime=time;}
  function pointAt(point,actor=avatar){actor.pointTarget=new THREE.Vector3(...point);actor.gesture='aim_target';}
  function setSpeech(value,actor=avatar){actor.speaking=Boolean(value);if(!actor.speaking)actor.visemes=null;}
  function setSpeechLevel(value,actor=avatar){actor.speechLevel=clamp(value);}
  // Phoneme-timed mouth weights, e.g. {viseme_aa:.8,jawOpen:.2}; null returns to
  // the amplitude envelope. Weights are only rendered while speaking.
  function setVisemes(weights,actor=avatar){actor.visemes=weights&&typeof weights==='object'?{...weights}:null;}
  function setFaceChannels(channels,actor=avatar){actor.face=channels;}
  function setIdle(value=true,actor=avatar){
    const current=actor.idle||{};
    actor.idle=typeof value==='object'&&value?{...current,...value}:{blink:Boolean(value),breath:Boolean(value),head:Boolean(value)};
    if(!actor.idle.blink)actor.blinkValue=0;
    return {...actor.idle};
  }
  function blink(actor=avatar){actor.blinkStart=clock.elapsedTime;actor.blinkNext=clock.elapsedTime+2+Math.random()*3;}
  const isActor=value=>Boolean(value&&typeof value==='object'&&value.root?.isObject3D&&'reach' in value);
  // Two-bone IK: reachTo('left'|'right'|'auto', point|Object3D|null, {weight,pole}).
  // Short form: reachTo(point|null, actor) = reachTo('auto', point, {}, actor).
  function reachTo(side,point,options={},actor=avatar){
    if(side!=='left'&&side!=='right'&&side!=='auto'&&side!==undefined&&side!==''){
      actor=isActor(point)?point:avatar;point=side;side='auto';options={};
    }
    if(isActor(options)){actor=options;options={};}
    const {weight=1,pole=null,immediate=false}=options||{};
    const auto=side==='auto'||!side;
    if(auto){
      if(point==null){for(const s of ['left','right'])reachTo(s,null,{immediate},actor);return null;}
      const p=resolvePoint(point,camera);
      side=p&&actor.root.worldToLocal(p.clone()).x<0?'right':'left';
      // A per-frame caller may switch sides; release the other hand smoothly.
      const other=side==='left'?'right':'left';if(actor.reach[other]?.auto)actor.reach[other].weight=0;
    }
    side=side==='right'?'right':'left';
    const entry=actor.reach[side];
    if(point==null){if(entry){entry.weight=0;if(immediate)delete actor.reach[side];}return null;}
    actor.reach[side]={target:point,pole,weight:clamp(weight),blend:immediate?clamp(weight):(entry?.blend||0),auto};
    return side;
  }
  function clearReach(actor=avatar){actor.reach={};}
  // Head/neck gaze toward a world point, an Object3D (followed each frame) or 'camera'.
  // Short form: lookAt(point|null, actor) uses weight 1.
  function lookAt(target,weight=1,actor=avatar){
    if(typeof weight!=='number'){actor=isActor(weight)?weight:actor;weight=1;}
    if(target==null){if(actor.gaze)actor.gaze.weight=0;return;}
    actor.gaze={target,weight:clamp(weight),blend:actor.gaze?.blend||0,point:actor.gaze?.point||null};
  }
  function setFacing(direction,actor=avatar){
    let angle=direction;
    if(typeof direction!=='number'){
      const p=resolvePoint(direction,camera);if(!p)return null;
      const origin=worldPosition(actor.root);angle=Math.atan2(p.x-origin.x,p.z-origin.z);
    }
    actor.root.rotation.y=angle;return angle;
  }
  function placeRoot(position,actor){
    if(!position)return;
    const p=Array.isArray(position)?(position.length>=3?new THREE.Vector3(position[0],0,position[2]):new THREE.Vector3(position[0],0,position[1])):resolvePoint(position,camera);
    if(p){actor.root.position.x=p.x;actor.root.position.z=p.z;}
  }
  // Seated posture: lowers the whole character so the hips rest at seatHeight
  // (world metres) and solves both legs to plant the feet on the floor. It holds
  // while recorded/BEAT clips drive the upper body. Short forms: sit(metres,
  // actor) keeps the current position and facing; sit(null, actor) stands.
  function sit(options={},actor=avatar){
    if(isActor(options)){actor=options;options={};}
    if(options===null)return stand(actor);
    if(typeof options==='number'||typeof options==='string')options={seatHeight:Number(options)};
    const {seatHeight=.45,position=null,facing=null,footForward=null,floorHeight=0,handsOnLap=true}=options;
    placeRoot(position,actor);if(facing!=null)setFacing(facing,actor);
    actor.posture={type:'sit',seatHeight:Number.isFinite(Number(seatHeight))?Number(seatHeight):.45,footForward,floorHeight:Number(floorHeight)||0,handsOnLap};
    return actor.posture;
  }
  // Lying posture on a surface top: 'back', 'left' or 'right' side; head points
  // along headDirection (world XZ vector or yaw angle).
  function lie({surfaceHeight=0,side='back',headDirection=[-1,0,0],position=null}={},actor=avatar){
    placeRoot(position,actor);
    actor.posture={type:'lie',surfaceHeight:Number(surfaceHeight)||0,side,headDirection};
    return actor.posture;
  }
  function stand(actor=avatar){actor.posture={type:'stand'};return actor.posture;}
  function setPosture(posture,actor=avatar){actor.posture=posture==null||posture==='auto'?undefined:posture;return actor.posture;}
  function moveTo(x,z,duration=1,actor=avatar){actor.move={start:clock.elapsedTime,from:actor.root.position.clone(),to:new THREE.Vector3(x,0,z),duration:Math.max(.01,duration)};}
  function addProp(name,x,z,color=0xe0b572,size=[.55,.65,.55]){if(props.has(name))scene.remove(props.get(name));const o=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshStandardMaterial({color}));o.position.set(x,size[1]/2,z);scene.add(o);props.set(name,o);return o;}
  const resize=new ResizeObserver(()=>{const w=Math.max(1,container.clientWidth),h=Math.max(260,container.clientHeight);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();});resize.observe(container);

  function activePosture(a){
    if(a.posture)return a.posture.type==='stand'?null:a.posture;
    const g=String(a.gesture||'').toLowerCase();
    if(AUTO_SIT.test(g))return {type:'sit',seatHeight:.45,floorHeight:0,handsOnLap:true,auto:true};
    if(AUTO_LIE.test(g))return {type:'lie',surfaceHeight:0,side:'back',headDirection:[-1,0,0],auto:true};
    return null;
  }
  function applyPostureTransform(a,posture){
    const rig=a.rig,{base,box,hipsY,height}=rig.metrics;
    rig.model.position.copy(base.position);rig.model.quaternion.copy(base.quaternion);
    if(posture?.type==='sit'){
      // The ischial contact sits roughly 0.1 m (scaled by stature) below the hip joints.
      const rootY=worldPosition(a.root).y;
      rig.model.position.y+=posture.seatHeight-rootY+.1*height/1.75-hipsY;
    }else if(posture?.type==='lie'){
      let yaw=posture.headDirection;
      if(typeof yaw!=='number'){
        const d=resolvePoint(Array.isArray(yaw)&&yaw.length===2?[yaw[0],0,yaw[1]]:yaw)||new THREE.Vector3(-1,0,0);
        d.applyQuaternion(a.root.getWorldQuaternion(new THREE.Quaternion()).invert());yaw=Math.atan2(-d.x,-d.z);
      }
      const roll=posture.side==='left'?-Math.PI/2:posture.side==='right'?Math.PI/2:0;
      const q=new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI/2,0,0)).premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),roll)).premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),yaw));
      const corners=[];for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z])corners.push(new THREE.Vector3(x,y,z).applyQuaternion(q));
      const rotated=new THREE.Box3().setFromPoints(corners),center=rotated.getCenter(new THREE.Vector3());
      rig.model.quaternion.premultiply(q);
      rig.model.position.applyQuaternion(q).add(new THREE.Vector3(-center.x,posture.surfaceHeight-worldPosition(a.root).y-rotated.min.y,-center.z));
    }
    rig.model.updateMatrixWorld(true);
  }
  function applyIdle(a,t){
    const idle=a.idle,rig=a.rig;if(!idle||a.motionActive)return;
    const basis=rig.model.getWorldQuaternion(new THREE.Quaternion()),up=new THREE.Vector3(0,1,0).applyQuaternion(basis),right=new THREE.Vector3(1,0,0).applyQuaternion(basis),forward=new THREE.Vector3(0,0,1).applyQuaternion(basis);
    if(idle.breath){
      const breath=Math.sin(t*Math.PI*2/4.2+a.phase);
      const [chest]=overlay(a,rig.bones.has('upperchest')?'upperchest':'chest');
      if(chest)rotateBoneWorld(chest,right,-.012*breath);
      const lift=.014*Math.max(0,breath);
      const [left]=overlay(a,'leftshoulder'),[rightShoulder]=overlay(a,'rightshoulder');
      if(left)rotateBoneWorld(left,forward,lift);
      if(rightShoulder)rotateBoneWorld(rightShoulder,forward,-lift);
    }
    if(idle.head){
      const scale=1-.7*(a.gaze?.blend||0),p=a.phase;
      const [head]=overlay(a,'head');
      if(head){
        rotateBoneWorld(head,up,scale*(.03*Math.sin(t*.29+p)+.012*Math.sin(t*.83+2*p)));
        rotateBoneWorld(head,right,scale*(.018*Math.sin(t*.37+1.7*p)+.008*Math.sin(t*1.13+p)));
      }
    }
  }
  function armPole(a,side){
    const rig=a.rig,basis=rig.model.getWorldQuaternion(new THREE.Quaternion()),shoulder=worldPosition(rig.bones.get(side+'arm')),s=rig.metrics.height/1.75;
    return shoulder.add(new THREE.Vector3(side==='left'?.3:-.3,-.6,-.35).multiplyScalar(s).applyQuaternion(basis));
  }
  function applyLegs(a,posture){
    if(posture?.type!=='sit')return;
    const rig=a.rig;
    // Recorded/BEAT clips fit a standing pelvis and legs; the seat owns them.
    // The pelvis returns to its bind orientation, the clip keeps the upper body.
    // The spine is counter-rotated so the clip's upper-body world pose is kept.
    if(a.motionActive){
      const [hips]=overlay(a,'hips'),[spine]=overlay(a,'spine');
      if(hips){
        const clip=hips.quaternion.clone(),bind=rig.bind.get(hips);
        hips.quaternion.copy(bind);
        if(spine?.parent===hips)spine.quaternion.premultiply(bind.clone().invert().multiply(clip));
        hips.updateMatrixWorld(true);
      }
    }
    const basis=rig.model.getWorldQuaternion(new THREE.Quaternion()),forward=new THREE.Vector3(0,0,1).applyQuaternion(basis),floorY=posture.floorHeight+rig.metrics.ankleY;
    for(const side of ['left','right']){
      const [thigh,knee]=overlay(a,side+'upleg',side+'leg'),foot=rig.bones.get(side+'foot');
      if(!thigh||!knee||!foot)continue;
      const hip=worldPosition(thigh),l1=hip.distanceTo(worldPosition(knee));
      const target=hip.clone().addScaledVector(forward,posture.footForward??l1*.95);target.y=floorY;
      solveTwoBoneIK(thigh,knee,foot,target,hip.clone().addScaledVector(forward,2).add(new THREE.Vector3(0,.6,0)),1);
    }
    if(posture.handsOnLap&&!a.motionActive&&/^(idle|sit|sitting|seated|sit_down|)$/.test(String(a.gesture||'').toLowerCase())){
      for(const side of ['left','right']){
        if(a.reach[side])continue;
        const thigh=rig.bones.get(side+'upleg'),knee=rig.bones.get(side+'leg'),[upper,lower]=overlay(a,side+'arm',side+'forearm'),hand=rig.bones.get(side+'hand');
        if(!thigh||!knee||!upper||!lower||!hand)continue;
        const lap=worldPosition(thigh).lerp(worldPosition(knee),.55).add(new THREE.Vector3(0,.09*rig.metrics.height/1.75,0));
        solveTwoBoneIK(upper,lower,hand,lap,armPole(a,side),.9);
      }
    }
  }
  function applyReach(a,dt){
    const rate=1-Math.exp(-dt*8);
    for(const side of ['left','right']){
      const r=a.reach[side];if(!r)continue;
      r.blend+=(r.weight-r.blend)*rate;
      if(r.weight===0&&r.blend<.01){delete a.reach[side];continue;}
      const target=resolvePoint(r.target,camera);if(!target)continue;
      const [upper,lower]=overlay(a,side+'arm',side+'forearm'),hand=a.rig.bones.get(side+'hand');
      solveTwoBoneIK(upper,lower,hand,target,resolvePoint(r.pole,camera)||armPole(a,side),r.blend);
    }
  }
  function applyGaze(a,dt){
    const g=a.gaze;if(!g)return;
    g.blend+=(g.weight-g.blend)*(1-Math.exp(-dt*6));
    if(g.weight===0&&g.blend<.01){a.gaze=null;return;}
    const target=resolvePoint(g.target,camera);if(!target)return;
    g.point=g.point?g.point.lerp(target,1-Math.exp(-dt*10)):target;
    const rig=a.rig,head=rig.bones.get('head');if(!head)return;
    const basis=rig.model.getWorldQuaternion(new THREE.Quaternion()),inverse=basis.clone().invert(),up=new THREE.Vector3(0,1,0).applyQuaternion(basis);
    const turn=(bone,fraction,maxYaw,maxPitch)=>{
      const local=rig.metrics.forwardLocal.get(bone);if(!local||fraction<=0)return;
      const eye=worldPosition(head).addScaledVector(up,.07*rig.metrics.height/1.75);
      const current=()=>local.clone().applyQuaternion(bone.getWorldQuaternion(new THREE.Quaternion()));
      const d=g.point.clone().sub(eye).normalize().applyQuaternion(inverse),c=current().applyQuaternion(inverse);
      let yaw=Math.atan2(d.x,d.z)-Math.atan2(c.x,c.z);yaw=Math.atan2(Math.sin(yaw),Math.cos(yaw));
      const pitch=Math.asin(Math.max(-1,Math.min(1,d.y)))-Math.asin(Math.max(-1,Math.min(1,c.y)));
      rotateBoneWorld(bone,up,Math.max(-maxYaw,Math.min(maxYaw,yaw*fraction)));
      const axis=new THREE.Vector3().crossVectors(current(),up);
      if(axis.lengthSq()>1e-8)rotateBoneWorld(bone,axis,Math.max(-maxPitch,Math.min(maxPitch,pitch*fraction)));
    };
    const [neck]=overlay(a,'neck');overlay(a,'head');
    const neckShare=neck?.4*g.blend:0;
    if(neck)turn(neck,neckShare,.5,.3);
    turn(head,neckShare<1?(g.blend-neckShare)/(1-neckShare):0,.8,.45);
  }
  function updateBlink(a,t){
    const face=a.face||{};
    if(!a.idle?.blink||'eyeBlinkLeft' in face||'eyeBlinkRight' in face){a.blinkValue=0;return;}
    if(a.blinkNext==null)a.blinkNext=t+1+Math.random()*3;
    if(t>=a.blinkNext){
      a.blinkStart=t;
      // Wide-eyed expressions (surprise, fear) blink less often.
      const wide=Math.max(a.expressionCurrent?.eyeWideLeft||0,a.expressionCurrent?.eyeWideRight||0)>.4;
      a.blinkNext=t+(Math.random()<.15?.32:(2.2+Math.random()*3.8)*(wide?1.7:1));
    }
    a.blinkValue=blinkCurve(t-a.blinkStart);
  }
  function updateExpression(a,dt){
    const target=expressionChannels(a.emotion,a.intensity),current=a.expressionCurrent||{},next={},rate=dt>0?1-Math.exp(-dt*10):1;
    for(const channel of new Set([...Object.keys(current),...Object.keys(target)])){
      const value=(current[channel]||0)+((target[channel]||0)-(current[channel]||0))*rate;
      if(value>1e-3)next[channel]=value;
    }
    a.expressionCurrent=next;
  }
  function animate(){if(disposed)return;const t=clock.getElapsedTime(),dt=Math.min(.1,Math.max(0,t-lastTime));lastTime=t;for(const a of actors){
    if(a.move){const k=Math.min(1,(t-a.move.start)/a.move.duration);a.root.position.lerpVectors(a.move.from,a.move.to,k);a.walking=k<1;if(k>=1)a.move=null;}else a.walking=/walk|move/.test(a.gesture);
    updateExpression(a,dt);updateBlink(a,t);
    const f=composeFaceChannels(a);
    if(a.rig){
      restoreOverlay(a);
      const posture=activePosture(a);
      applyPostureTransform(a,posture);
      if(!a.motionActive){
        applyGesturePose(a.rig,a.gesture,a.gestureTime??t);
        if(a.pointTarget&&/aim_target|point|reach/.test(a.gesture)){
          const side=a.pointTarget.x>=a.root.getWorldPosition(new THREE.Vector3()).x?'left':'right';
          for(const [parent,child] of [[side+'arm',side+'forearm'],[side+'forearm',side+'hand']]){
            const bone=a.rig.bones.get(parent),end=a.rig.bones.get(child);
            if(bone&&end)aimBoneToward(bone,end,a.pointTarget.clone().sub(bone.getWorldPosition(new THREE.Vector3())));
          }
        }
      }
      a.overlayBase=new Map();a.rig.model.updateMatrixWorld(true);
      applyIdle(a,t);applyLegs(a,posture);applyReach(a,dt);applyGaze(a,dt);
      const values=new Map();for(const [name,value] of Object.entries(f)){const n=morphName(name);if(!values.has(n))values.set(n,value);}
      for(const mesh of a.rig.morphs){
        const defaults=a.rig.morphDefaults.get(mesh);
        for(const [target,index] of a.rig.morphKeys.get(mesh))mesh.morphTargetInfluences[index]=values.has(target)?clamp(values.get(target)):defaults[index];
      }
    }else{
      a.root.position.y=.012*Math.sin(t*1.6);a.head.rotation.y=.04*Math.sin(t*.6);
      a.legs.forEach((l,i)=>l.rotation.x=a.walking?.3*Math.sin(t*8+i*Math.PI):0);
      a.arms[0].rotation.set(0,0,.08);a.arms[1].rotation.set(0,0,-.08);
      if(/point|touch|reach/.test(a.gesture)){a.arms[1].rotation.x=-1.2;a.arms[1].rotation.z=-.35;}
      // Open palm: straight stick arms swing forward, hands in front of the body at waist height, shoulder width apart.
      else if(/welcome|open|explain|offer|reassure|present/.test(a.gesture)){a.arms[0].rotation.z=-.05;a.arms[1].rotation.z=.05;a.arms.forEach(l=>l.rotation.x=-.9);}
      else if(/think/.test(a.gesture))a.arms[1].rotation.x=-2.2;
      else if(/wave|beat/.test(a.gesture))a.arms[1].rotation.z=-1.6+.25*Math.sin(t*5);
      a.mouth.scale.y=clamp(f.jawOpen)*.8+.15;
      a.brows.forEach((b,i)=>{const up=(f.browInnerUp||0)+(f[i?'browOuterUpRight':'browOuterUpLeft']||0),down=f[i?'browDownRight':'browDownLeft']||0;b.position.y=.092+up*.025-down*.03;b.rotation.z=(i?1:-1)*down*.3;});
      a.eyes.forEach((e,i)=>{const wide=f[i?'eyeWideRight':'eyeWideLeft']||0,squint=f[i?'eyeSquintRight':'eyeSquintLeft']||0,closed=f[i?'eyeBlinkRight':'eyeBlinkLeft']||0;e.scale.y=Math.max(.08,(1+wide*.8-squint*.8)*(1-.9*closed));});
    }
  }renderer.render(scene,camera);raf=requestAnimationFrame(animate);}
  animate();
  return {scene,camera,renderer,avatar,actors,ready,makeAvatar,setCharacter,gesture,pointAt,expression,setExpression,setSpeech,setSpeechLevel,setVisemes,setFaceChannels,setIdle,blink,reachTo,clearReach,lookAt,gaze:lookAt,sit,lie,stand,setPosture,setFacing,makeOccluder,setSkeleton,setMotionFrame,setPosePositions,clearMotion,moveTo,addProp,showAvatar(){avatar.root.visible=true;skeleton.visible=false;},capture(){renderer.render(scene,camera);return renderer.domElement.toDataURL('image/png');},dispose(){disposed=true;cancelAnimationFrame(raf);resize.disconnect();status.remove();selector.remove();if(!previousPosition)container.style.position='';disposeObject(scene);renderer.dispose();renderer.domElement.remove();}};
}
