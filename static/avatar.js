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
function channelsFor(actor,t) {
  const f={...(actor.face||{})},n=actor.emotion,i=actor.intensity;
  if (['joy','happy'].includes(n)){f.mouthSmileLeft=Math.max(f.mouthSmileLeft||0,.7*i);f.mouthSmileRight=Math.max(f.mouthSmileRight||0,.7*i);f.cheekSquintLeft=Math.max(f.cheekSquintLeft||0,.4*i);f.cheekSquintRight=Math.max(f.cheekSquintRight||0,.4*i);}
  if (['surprise'].includes(n)){f.browInnerUp=Math.max(f.browInnerUp||0,.8*i);f.eyeWideLeft=Math.max(f.eyeWideLeft||0,.6*i);f.eyeWideRight=Math.max(f.eyeWideRight||0,.6*i);f.jawOpen=Math.max(f.jawOpen||0,.45*i);}
  if (['anger','angry'].includes(n)){f.browDownLeft=Math.max(f.browDownLeft||0,.65*i);f.browDownRight=Math.max(f.browDownRight||0,.65*i);}
  if (['sad','sadness'].includes(n)){f.browInnerUp=Math.max(f.browInnerUp||0,.5*i);f.mouthFrownLeft=Math.max(f.mouthFrownLeft||0,.5*i);f.mouthFrownRight=Math.max(f.mouthFrownRight||0,.5*i);}
  if (actor.speaking) {
    // This is an approximate animated speaking envelope, not phoneme alignment.
    const pulse=.25+.45*Math.abs(Math.sin(t*12));
    f.jawOpen=Math.max(f.jawOpen||0,pulse); f.mouthFunnel=Math.max(f.mouthFunnel||0,.22*Math.abs(Math.sin(t*7)));
    f.viseme_aa=Math.max(f.viseme_aa||0,pulse*.55);
    f.viseme_O=Math.max(f.viseme_O||0,.18*Math.abs(Math.sin(t*7)));
  }
  return f;
}

export function createStage(container, options={}) {
  const scene=new THREE.Scene();scene.background=new THREE.Color(options.background||'#101827');
  const camera=new THREE.PerspectiveCamera(42,1,.01,100);camera.position.set(0,1.6,5.8);camera.lookAt(0,1.15,0);
  const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));container.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xe6f4ff,0x344055,2.4));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(3,5,4);scene.add(light);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.MeshStandardMaterial({color:0x182438,roughness:.9}));floor.rotation.x=-Math.PI/2;scene.add(floor);
  const grid=new THREE.GridHelper(12,24,0x405575,0x24344b);grid.position.y=.002;scene.add(grid);
  const actors=[],props=new Map(),clock=new THREE.Clock(),loader=new GLTFLoader();let raf,disposed=false;
  const modelChoices={rowan:new URL('./avatars/rowan.glb',import.meta.url).href,mira:new URL('./avatars/mira.glb',import.meta.url).href};
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
    const actor={root,procedural,head,arms,legs,mouth,eyes,brows,gesture:'idle',emotion:'neutral',intensity:.5,speaking:false,walking:false,face:null,move:null,rig:null,loadError:null};
    actors.push(actor);
    const request=actor.loadRequest=1;
    actor.ready=avatarUrl?new Promise(resolve=>loader.load(avatarUrl,gltf=>{
      if(disposed){disposeObject(gltf.scene);resolve({loaded:false,error:new Error('Stage disposed')});return;}
      if(request!==actor.loadRequest){disposeObject(gltf.scene);resolve({loaded:false,stale:true,actor});return;}
      const model=gltf.scene;root.add(model);model.updateMatrixWorld(true);
      const rig={model,bones:new Map(),bind:new Map(),restWorld:new Map(),morphs:[],morphDefaults:new Map()};
      model.traverse(o=>{
        if(o.isBone){const name=key(o.name);if(!rig.bones.has(name))rig.bones.set(name,o);rig.bind.set(o,o.quaternion.clone());}
        if(o.isMesh){o.frustumCulled=false;if(o.morphTargetDictionary&&o.morphTargetInfluences){rig.morphs.push(o);rig.morphDefaults.set(o,o.morphTargetInfluences.slice());}}
      });
      for(const bone of rig.bind.keys())rig.restWorld.set(bone,bone.getWorldQuaternion(new THREE.Quaternion()));
      if(actor.rig){root.remove(actor.rig.model);disposeObject(actor.rig.model);}
      actor.rig=rig;actor.loadError=null;procedural.visible=false;resolve({loaded:true,actor});
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
      const model=gltf.scene;actor.root.add(model);model.updateMatrixWorld(true);
      const rig={model,bones:new Map(),bind:new Map(),restWorld:new Map(),morphs:[],morphDefaults:new Map()};
      model.traverse(o=>{if(o.isBone){const name=key(o.name);if(!rig.bones.has(name))rig.bones.set(name,o);rig.bind.set(o,o.quaternion.clone());}if(o.isMesh){o.frustumCulled=false;if(o.morphTargetDictionary&&o.morphTargetInfluences){rig.morphs.push(o);rig.morphDefaults.set(o,o.morphTargetInfluences.slice());}}});
      for(const bone of rig.bind.keys())rig.restWorld.set(bone,bone.getWorldQuaternion(new THREE.Quaternion()));
      if(actor.rig){actor.root.remove(actor.rig.model);disposeObject(actor.rig.model);}actor.rig=rig;actor.procedural.visible=false;actor.loadError=null;resolve({loaded:true,actor});
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
  function setMotionFrame(frame,actor=avatar,source={}){
    if(!actor.rig||!frame)return false;
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
    if(Array.isArray(frame.positions))setPosePositions(frame.positions,names,actor,{basis:source.restBasis,axisSigns:signs});
    actor.motionActive=applied>0;
    if(frame.rootTranslation){const p=new THREE.Vector3(...frame.rootTranslation.map((v,i)=>(Number(v)||0)*signs[i])).applyQuaternion(basis);actor.root.position.copy(p);}
    return applied>0;
  }
  function setPosePositions(joints,jointNames,actor=avatar,options={}){
    if(!actor.rig||!Array.isArray(joints)||!Array.isArray(jointNames))return false;
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
  function clearMotion(actor=avatar){actor.motionActive=false;if(actor.rig)for(const [bone,bind] of actor.rig.bind)bone.quaternion.copy(bind);}
  function expression(name='neutral',intensity=.5,actor=avatar){actor.emotion=name;actor.intensity=clamp(intensity);}
  function gesture(name='idle',actor=avatar){actor.gesture=name;}
  function setSpeech(value,actor=avatar){actor.speaking=Boolean(value);}
  function setFaceChannels(channels,actor=avatar){actor.face=channels;}
  function moveTo(x,z,duration=1,actor=avatar){actor.move={start:clock.elapsedTime,from:actor.root.position.clone(),to:new THREE.Vector3(x,0,z),duration:Math.max(.01,duration)};}
  function addProp(name,x,z,color=0xe0b572,size=[.55,.65,.55]){if(props.has(name))scene.remove(props.get(name));const o=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshStandardMaterial({color}));o.position.set(x,size[1]/2,z);scene.add(o);props.set(name,o);return o;}
  const resize=new ResizeObserver(()=>{const w=Math.max(1,container.clientWidth),h=Math.max(260,container.clientHeight);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();});resize.observe(container);
  function animate(){if(disposed)return;const t=clock.getElapsedTime();for(const a of actors){
    if(a.move){const k=Math.min(1,(t-a.move.start)/a.move.duration);a.root.position.lerpVectors(a.move.from,a.move.to,k);a.walking=k<1;if(k>=1)a.move=null;}else a.walking=/walk|move/.test(a.gesture);
    const f=channelsFor(a,t);
    if(a.rig){
      if(!a.motionActive){
        for(const [bone,bind] of a.rig.bind)bone.quaternion.copy(bind);
        const rotate=(name,x=0,y=0,z=0)=>{const bone=a.rig.bones.get(name);if(bone)bone.quaternion.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(x,y,z)));};
        if(a.walking){const swing=.25*Math.sin(t*8);rotate('leftupleg',swing);rotate('rightupleg',-swing);rotate('leftarm',-swing*.6);rotate('rightarm',swing*.6);}
        if(/point|touch|reach/.test(a.gesture))rotate('rightarm',-1.1,0,-.25);
        else if(/welcome|open|explain/.test(a.gesture)){rotate('leftarm',-.25,0,.55);rotate('rightarm',-.25,0,-.55);}
        else if(/think/.test(a.gesture))rotate('rightarm',-1.4);
        else if(/wave|beat/.test(a.gesture))rotate('rightarm',-.5,0,-1+.25*Math.sin(t*5));
      }
      for(const mesh of a.rig.morphs)for(const [name,index] of Object.entries(mesh.morphTargetDictionary)){
        const target=morphName(name);
        if(!/^(brow|cheek|eye|jaw|mouth|nose|tongue|viseme)/.test(target))continue;
        const entry=Object.entries(f).find(([k])=>morphName(k)===target);
        mesh.morphTargetInfluences[index]=entry?clamp(entry[1]):a.rig.morphDefaults.get(mesh)[index];
      }
    }else{
      a.root.position.y=.012*Math.sin(t*1.6);a.head.rotation.y=.04*Math.sin(t*.6);
      a.legs.forEach((l,i)=>l.rotation.x=a.walking?.3*Math.sin(t*8+i*Math.PI):0);
      a.arms[0].rotation.set(0,0,.08);a.arms[1].rotation.set(0,0,-.08);
      if(/point|touch|reach/.test(a.gesture)){a.arms[1].rotation.x=-1.2;a.arms[1].rotation.z=-.35;}
      else if(/welcome|open|explain/.test(a.gesture)){a.arms[0].rotation.z=.65;a.arms[1].rotation.z=-.65;a.arms.forEach(l=>l.rotation.x=-.25);}
      else if(/think/.test(a.gesture))a.arms[1].rotation.x=-2.2;
      else if(/wave|beat/.test(a.gesture))a.arms[1].rotation.z=-1.6+.25*Math.sin(t*5);
      a.mouth.scale.y=clamp(f.jawOpen)*.8+.15;
      a.brows.forEach((b,i)=>{const up=(f.browInnerUp||0)+(f[i?'browOuterUpRight':'browOuterUpLeft']||0),down=f[i?'browDownRight':'browDownLeft']||0;b.position.y=.092+up*.025-down*.03;b.rotation.z=(i?1:-1)*down*.3;});
      a.eyes.forEach((e,i)=>{const wide=f[i?'eyeWideRight':'eyeWideLeft']||0,squint=f[i?'eyeSquintRight':'eyeSquintLeft']||0;e.scale.y=Math.max(.15,1+wide*.8-squint*.8);});
    }
  }renderer.render(scene,camera);raf=requestAnimationFrame(animate);}
  animate();
  return {scene,camera,renderer,avatar,actors,ready,makeAvatar,setCharacter,gesture,expression,setSpeech,setFaceChannels,setSkeleton,setMotionFrame,setPosePositions,clearMotion,moveTo,addProp,showAvatar(){avatar.root.visible=true;skeleton.visible=false;},capture(){renderer.render(scene,camera);return renderer.domElement.toDataURL('image/png');},dispose(){disposed=true;cancelAnimationFrame(raf);resize.disconnect();status.remove();selector.remove();if(!previousPosition)container.style.position='';disposeObject(scene);renderer.dispose();renderer.domElement.remove();}};
}

