import * as THREE from './vendor/three.module.js';

// Original procedural character and motion-data viewer; no downloaded avatar assets.
export function createStage(container, options = {}) {
  const scene = new THREE.Scene(); scene.background = new THREE.Color(options.background || '#101827');
  const camera = new THREE.PerspectiveCamera(42, 1, .01, 100); camera.position.set(0, 1.6, 5.8); camera.lookAt(0, 1.15, 0);
  const renderer = new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2)); container.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xe6f4ff,0x344055,2.4));
  const light=new THREE.DirectionalLight(0xffffff,2); light.position.set(3,5,4); scene.add(light);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.MeshStandardMaterial({color:0x182438,roughness:.9})); floor.rotation.x=-Math.PI/2; scene.add(floor);
  const grid=new THREE.GridHelper(12,24,0x405575,0x24344b);grid.position.y=.002;scene.add(grid);
  const actors=[],props=new Map(),clock=new THREE.Clock();let raf,disposed=false;
  function makeAvatar(color=0x60c8d9,x=0,z=0) {
    const root=new THREE.Group(); root.position.set(x,0,z);scene.add(root);
    const mat=new THREE.MeshStandardMaterial({color,roughness:.65}),skin=new THREE.MeshStandardMaterial({color:0xe1ad91,roughness:.8}),dark=new THREE.MeshStandardMaterial({color:0x182030});
    function mesh(geometry,material,parent=root,pos=[0,0,0]){const m=new THREE.Mesh(geometry,material);m.position.set(...pos);parent.add(m);return m;}
    mesh(new THREE.CapsuleGeometry(.25,.5,5,12),mat,root,[0,1.15,0]);
    const head=new THREE.Group();head.position.y=1.88;root.add(head);mesh(new THREE.SphereGeometry(.24,24,20),skin,head);
    const eyes=[-.085,.085].map(x=>mesh(new THREE.SphereGeometry(.034,12,8),dark,head,[x,.025,.218]));
    const brows=[-.085,.085].map(x=>mesh(new THREE.BoxGeometry(.105,.014,.015),dark,head,[x,.092,.217]));
    const mouth=mesh(new THREE.SphereGeometry(.06,16,8),dark,head,[0,-.085,.225]);mouth.scale.set(1,.15,.25);
    function limb(x,y,length){const pivot=new THREE.Group();pivot.position.set(x,y,0);root.add(pivot);mesh(new THREE.CapsuleGeometry(.065,length,4,8),mat,pivot,[0,-length/2,0]);return pivot;}
    const arms=[limb(-.32,1.52,.58),limb(.32,1.52,.58)],legs=[limb(-.13,.76,.65),limb(.13,.76,.65)];
    const state={root,head,arms,legs,mouth,eyes,brows,gesture:'idle',emotion:'neutral',intensity:.5,speaking:false,walking:false,face:null,move:null};actors.push(state);return state;
  }
  const avatar=makeAvatar(options.color||0x60c8d9);
  const skeleton=new THREE.Group();scene.add(skeleton);let points=[],bones=[];
  function setSkeleton(joints,edges=[]) {
    if(!joints?.length)return;
    skeleton.visible=true;
    const vectors=joints.map(p=>new THREE.Vector3(p[0]||0,p[1]||0,p[2]||0));
    if(points.length!==vectors.length){for(const o of [...points,...bones]){skeleton.remove(o);o.geometry.dispose();o.material.dispose();}points=[];bones=[];for(const v of vectors){const o=new THREE.Mesh(new THREE.SphereGeometry(.035,8,6),new THREE.MeshBasicMaterial({color:0xa9e879}));skeleton.add(o);points.push(o);}}
    points.forEach((o,i)=>o.position.copy(vectors[i]));
    while(bones.length<edges.length){const o=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x81cee5}));skeleton.add(o);bones.push(o);}
    bones.forEach((o,i)=>{o.visible=i<edges.length;if(o.visible)o.geometry.setFromPoints([vectors[edges[i][0]],vectors[edges[i][1]]]);});
    avatar.root.visible=false;
  }
  function expression(name='neutral',intensity=.5,actor=avatar){actor.emotion=name;actor.intensity=Math.max(0,Math.min(1,Number(intensity)||0));}
  function gesture(name='idle',actor=avatar){actor.gesture=name;}
  function setSpeech(value,actor=avatar){actor.speaking=Boolean(value);}
  function setFaceChannels(channels,actor=avatar){actor.face=channels;}
  function moveTo(x,z,duration=1,actor=avatar){actor.move={start:clock.elapsedTime,from:actor.root.position.clone(),to:new THREE.Vector3(x,0,z),duration:Math.max(.01,duration)};}
  function addProp(name,x,z,color=0xe0b572,size=[.55,.65,.55]){if(props.has(name))scene.remove(props.get(name));const o=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshStandardMaterial({color}));o.position.set(x,size[1]/2,z);scene.add(o);props.set(name,o);return o;}
  const resize=new ResizeObserver(()=>{const w=Math.max(1,container.clientWidth),h=Math.max(260,container.clientHeight);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();});resize.observe(container);
  function animate(){if(disposed)return;const t=clock.getElapsedTime();for(const a of actors){
    a.root.position.y=.012*Math.sin(t*1.6);a.head.rotation.y=.04*Math.sin(t*.6);
    if(a.move){const k=Math.min(1,(t-a.move.start)/a.move.duration);a.root.position.lerpVectors(a.move.from,a.move.to,k);a.walking=k<1;if(k>=1)a.move=null;}else a.walking=/walk|move/.test(a.gesture);
    a.legs.forEach((l,i)=>l.rotation.x=a.walking?.3*Math.sin(t*8+i*Math.PI):0);
    a.arms[0].rotation.set(0,0,.08);a.arms[1].rotation.set(0,0,-.08);
    if(/point|touch|reach/.test(a.gesture)){a.arms[1].rotation.x=-1.2;a.arms[1].rotation.z=-.35;}
    else if(/welcome|open|explain/.test(a.gesture)){a.arms[0].rotation.z=.65;a.arms[1].rotation.z=-.65;a.arms.forEach(l=>l.rotation.x=-.25);}
    else if(/think/.test(a.gesture)){a.arms[1].rotation.x=-2.2;}
    else if(/wave|beat/.test(a.gesture)){a.arms[1].rotation.z=-1.6+.25*Math.sin(t*5);}
    a.mouth.scale.y=a.speaking?.25+.45*Math.abs(Math.sin(t*12)):.15;
    const f=a.face||{};
    a.brows.forEach((b,i)=>{const up=(f.browInnerUp||0)+(f[i?'browOuterUpRight':'browOuterUpLeft']||0),down=f[i?'browDownRight':'browDownLeft']||0;b.position.y=.092+up*.025-down*.03+(['surprise','joy','happy'].includes(a.emotion)?.03*a.intensity:0);b.rotation.z=(i?1:-1)*(down*.3+(['anger','angry','sadness','sad'].includes(a.emotion)?.25*a.intensity:0));});
    a.eyes.forEach((e,i)=>{const wide=f[i?'eyeWideRight':'eyeWideLeft']||0,squint=f[i?'eyeSquintRight':'eyeSquintLeft']||0;e.scale.y=Math.max(.15,1+wide*.8-squint*.8);});
  }renderer.render(scene,camera);raf=requestAnimationFrame(animate);}
  animate();
  return {scene,camera,renderer,avatar,actors,makeAvatar,gesture,expression,setSpeech,setFaceChannels,setSkeleton,moveTo,addProp,showAvatar(){avatar.root.visible=true;skeleton.visible=false;},capture(){renderer.render(scene,camera);return renderer.domElement.toDataURL('image/png');},dispose(){disposed=true;cancelAnimationFrame(raf);resize.disconnect();scene.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});renderer.dispose();renderer.domElement.remove();}};
}
