import struct,json,zlib,time,os,zipfile,tarfile,binascii,shutil
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
ROOT=Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27')
OUT=ROOT/'outputs'; REP=ROOT/'reports'; TOOLS=ROOT/'tools'
if ROOT.exists(): shutil.rmtree(ROOT)
for d in [OUT,REP,TOOLS]: d.mkdir(parents=True,exist_ok=True)
SRC=Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/outputs/v25_attribute_group_render_container.3dphox')
V25_REPORT=Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json')
v25_report=json.loads(V25_REPORT.read_text())
with SRC.open('rb') as f:
    magic=f.read(11); ml=struct.unpack('<Q',f.read(8))[0]; man=json.loads(f.read(ml)); blob=f.read()
chunks={c['name']:c for c in man['chunks']}
def comp_slice(name):
    c=chunks[name]; return blob[c['offset']:c['offset']+c['compressed_bytes']]
def dec(name): return zlib.decompress(comp_slice(name))
N=v25_report['input']['source_splats']; source_ply=v25_report['input']['source_ply_bytes']; v11=v25_report['input']['v11_vq256_bytes']; v25_size=SRC.stat().st_size
print('decode SH', flush=True)
sh=np.frombuffer(dec('sh_rest_q8_global'),np.int8).reshape(N,45).copy()
global_scale=chunks['sh_rest_q8_global'].get('global_scale',0.006946287755891094)
rng=np.random.default_rng(27027); sample_idx=rng.choice(N, size=min(5000,N), replace=False)
K=128
labels=np.empty((N,3),np.uint8); codebooks=np.empty((3,K,15),np.int8); rmses=[]; max_abs=[]
for g in range(3):
    print('fit group',g,flush=True)
    X=sh[:,g*15:(g+1)*15].astype(np.float32)
    km=MiniBatchKMeans(n_clusters=K,n_init=1,max_iter=3,batch_size=2048,random_state=27027+g,reassignment_ratio=0.01)
    km.fit(X[sample_idx])
    lab=km.predict(X).astype(np.uint8)
    centers=np.clip(np.rint(km.cluster_centers_),-128,127).astype(np.int8)
    rec=centers[lab].astype(np.int16); err=X.astype(np.int16)-rec
    labels[:,g]=lab; codebooks[g]=centers
    rmses.append(float(np.sqrt(np.mean(err.astype(np.float64)**2)))); max_abs.append(int(np.max(np.abs(err))))
# Sample correction entropy estimate only, not full exact archive this cycle.
# Compose container: copy v25 non-SH compressed chunks and append new VQ chunks.
copy=['tier_labels_u8','xyz_u24_fixed','dc_rgb_opacity_u8','scale_f16','quat_i16_norm4']
new_chunks=[]; payloads=[]; offset=0
for name in copy:
    c=dict(chunks[name]); cs=comp_slice(name); c['offset']=offset; c['compressed_bytes']=len(cs); c['source']='v25'; new_chunks.append(c); payloads.append(cs); offset+=len(cs)
for name,raw,dtype,shape,semantic in [
    ('sh_vq128_idx_u8', labels.tobytes(), 'uint8', [N,3], 'product VQ label stream: one 8-bit label for each 15-coeff SH group'),
    ('sh_vq128_codebook_i8', codebooks.tobytes(), 'int8', [3,K,15], 'rounded int8 codebooks for SH VQ render core')]:
    comp=zlib.compress(raw,6); c={'name':name,'offset':offset,'raw_bytes':len(raw),'compressed_bytes':len(comp),'crc32_raw':binascii.crc32(raw)&0xffffffff,'dtype':dtype,'shape':shape,'semantic':semantic}; new_chunks.append(c); payloads.append(comp); offset+=len(comp)
manifest={'format':'CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V27_SH_VQ_RENDER','cycle':'v0.27','source':'v25 honest full-attribute container','source_splats':N,'source_ply_bytes':source_ply,'chunks':new_chunks,'sh_vq':{'groups':'3x15','k':K,'sample_train_rows':len(sample_idx),'rmse_q8_by_group':rmses,'rmse_float_estimate_by_group':[r*global_scale for r in rmses],'max_abs_q8_error_by_group':max_abs,'note':'This render container is not q8-exact. It is a size repair replacing the v25 global q8 SH stream with a palette/codebook render core. Exact q8 correction is deferred to v0.28.'},'truth_contract':{'carried_exact_from_v25':['tier labels','xyz_u24_fixed','dc_rgb_opacity_u8','scale_f16','quat_i16_norm4'],'changed':['sh_rest_q8_global -> sh_vq128_idx_u8 + sh_vq128_codebook_i8'],'not_claimed':'lossless SH or final render parity'}}
container=OUT/'v27_attribute_group_sh_vq_render_container.3dphox'
with container.open('wb') as f:
    m=json.dumps(manifest,indent=2).encode(); f.write(b'CRYPSOID27\0'); f.write(struct.pack('<Q',len(m))); f.write(m); [f.write(p) for p in payloads]
# verify
with container.open('rb') as f:
    f.read(11); ml=struct.unpack('<Q',f.read(8))[0]; mm=json.loads(f.read(ml)); bb=f.read()
verified=[]; all_ok=True
for c in mm['chunks']:
    raw=zlib.decompress(bb[c['offset']:c['offset']+c['compressed_bytes']]); ok=(binascii.crc32(raw)&0xffffffff)==c['crc32_raw']; all_ok&=ok; verified.append({'name':c['name'],'ok':ok,'compressed_bytes':c['compressed_bytes'],'raw_bytes':c['raw_bytes']})
render_size=container.stat().st_size
# SVG
bars=[('v11 VQ128 baseline',v11),('v25 q8 full attr',v25_size),('v27 SH-VQ full attr render',render_size)]
maxb=max(b for _,b in bars); svg=['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="210"><rect width="100%" height="100%" fill="white"/><style>text{font-family:monospace;font-size:14px}</style>']
y=45
for label,b in bars:
    w=560*b/maxb; svg.append(f'<text x="20" y="{y}">{label}</text><rect x="260" y="{y-15}" width="{w:.1f}" height="20" fill="#222"/><text x="{270+w:.1f}" y="{y}">{b/1024/1024:.2f} MiB</text>'); y+=50
svg.append('</svg>'); (OUT/'v27_sh_vq_size_bars.svg').write_text('\n'.join(svg))
report={'cycle':'v0.27','title':'SH debt breaker using product VQ render core','status':'Built actual .3dphox render container preserving v25 non-SH attribute groups and replacing the huge global q8 SH stream with VQ labels/codebooks. This fixes the v25 loop by attacking the largest honest chunk first.','source':{'splats':N,'source_ply_bytes':source_ply,'v11_vq256_bytes':v11,'v25_full_attribute_bytes':v25_size},'v25_debt':{'sh_q8_global_compressed_bytes':chunks['sh_rest_q8_global']['compressed_bytes'],'share_of_v25_percent':100*chunks['sh_rest_q8_global']['compressed_bytes']/v25_size},'v27':{'render_container_bytes':render_size,'render_container_mib':render_size/1024/1024,'ratio_vs_source_ply':source_ply/render_size,'reduction_vs_source_ply_percent':100*(1-render_size/source_ply),'delta_vs_v25_percent':100*(render_size/v25_size-1),'delta_vs_v11_percent':100*(render_size/v11-1)},'sh_vq':manifest['sh_vq'],'readback':{'all_crc_ok':all_ok,'verified_chunks':verified},'outputs':{'container':str(container),'bars':str(OUT/'v27_sh_vq_size_bars.svg')},'next':'v0.28 should add q8-exact correction chunks and make VQ context-aware by tier/phoxoid family; v0.27 is a render-size repair, not a lossless archive win.'}
(REP/'PHOXBENCH_V27_SH_DEBT_REPORT.json').write_text(json.dumps(report,indent=2))
(REP/'RESEARCH_BUILD_TEST_CYCLE_V27.md').write_text(f'''# CRYPSOID v0.27 — SH attribute debt breaker\n\nThis cycle attacks the real v0.25 loop: the honest full-attribute container got large because SH residuals were 41.89% of the file.\n\n## Result\n\n| Container | Size | Change |\n|---|---:|---:|\n| v11 VQ128 baseline | {v11/1024/1024:.2f} MiB | baseline |\n| v25 q8 full attribute container | {v25_size/1024/1024:.2f} MiB | +{100*(v25_size/v11-1):.2f}% vs v11 |\n| v27 SH-VQ full attribute render container | {render_size/1024/1024:.2f} MiB | {100*(render_size/v25_size-1):.2f}% vs v25 |\n\n## What is preserved\n\nCopied exactly from v25: tier labels, XYZ u24, DC/opacity u8, scale f16, quaternion i16.\n\nChanged: `sh_rest_q8_global` becomes `sh_vq128_idx_u8` + `sh_vq128_codebook_i8`.\n\n## Caveat\n\nThis is a render container, not an exact SH archive. V0.28 needs exact correction chunks and context-aware codebooks.\n\nCRC readback: `{all_ok}`.\n''')
shutil.copy(__file__, TOOLS/'build_v27_fast.py')
# packages stored/no extra compression to avoid download loop delays
zip_path=Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27.zip');
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_STORED) as z:
    for path in ROOT.rglob('*'):
        if path.is_file(): z.write(path,path.relative_to(ROOT.parent))
tar_path=Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27.tar.gz')
if tar_path.exists(): tar_path.unlink()
with tarfile.open(tar_path,'w:gz') as tar: tar.add(ROOT,arcname=ROOT.name)
print(json.dumps({'render_size':render_size,'all_crc_ok':all_ok,'zip':str(zip_path),'tar':str(tar_path)},indent=2), flush=True)
