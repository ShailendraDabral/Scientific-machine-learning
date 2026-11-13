import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { viridis, type View2D } from '../fields';

interface Stage {
  renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera;
  mesh: THREE.Mesh | null; dims: string; render: () => void;
}

/** Three.js surface of a 2D field. Heights and colours are updated in place as new snapshots arrive. */
export function Surface3D({ view, zRange, height = 300 }: { view: View2D | null; zRange: [number, number]; height?: number }) {
  const mount = useRef<HTMLDivElement>(null);
  const stage = useRef<Stage | null>(null);

  useEffect(() => {
    const el = mount.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, height);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, el.clientWidth / height, 0.1, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(1.9, -2.3, 1.5);
    scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(1, -1.5, 3);
    scene.add(sun);
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 1.4, 0.9)),
      new THREE.LineBasicMaterial({ color: 0x33445a }));
    scene.add(frame);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    const render = () => renderer.render(scene, camera);
    controls.addEventListener('change', render);
    const observer = new ResizeObserver(() => {
      renderer.setSize(el.clientWidth, height);
      camera.aspect = el.clientWidth / height;
      camera.updateProjectionMatrix();
      render();
    });
    observer.observe(el);
    stage.current = { renderer, scene, camera, mesh: null, dims: '', render };
    render();
    return () => {
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      stage.current = null;
    };
  }, [height]);

  useEffect(() => {
    const st = stage.current;
    if (!st || !view) return;
    const rows = view.z.length, cols = view.z[0].length;
    const dims = `${rows}x${cols}`;
    if (st.dims !== dims) {
      if (st.mesh) { st.scene.remove(st.mesh); st.mesh.geometry.dispose(); }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(rows * cols * 3), 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(rows * cols * 3), 3));
      const index: number[] = [];
      for (let r = 0; r < rows - 1; r++) for (let c = 0; c < cols - 1; c++) {
        const a = r * cols + c;
        index.push(a, a + 1, a + cols, a + 1, a + cols + 1, a + cols);
      }
      geometry.setIndex(index);
      st.mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      st.scene.add(st.mesh);
      st.dims = dims;
    }
    const pos = st.mesh!.geometry.getAttribute('position') as THREE.BufferAttribute;
    const col = st.mesh!.geometry.getAttribute('color') as THREE.BufferAttribute;
    const [lo, hi] = zRange;
    const span = hi - lo || 1;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const t = (view.z[r][c] - lo) / span;
      pos.setXYZ(i, -1 + (2 * c) / (cols - 1), -0.7 + (1.4 * r) / (rows - 1), -0.45 + 0.9 * Math.min(1, Math.max(0, t)));
      const [red, green, blue] = viridis(t);
      col.setXYZ(i, red, green, blue);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    st.mesh!.geometry.computeVertexNormals();
    st.render();
  }, [view, zRange]);

  return <div ref={mount} style={{ height }} className="w-full cursor-grab" data-testid="surface3d" />;
}
