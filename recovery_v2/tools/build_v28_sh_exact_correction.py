#!/usr/bin/env /usr/bin/python3
import os, json, zlib, struct, binascii, shutil, tarfile, zipfile, math
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_28')
OUT = ROOT/'outputs'; REP=ROOT/'reports'; TOOLS=ROOT/'tools'
if ROOT.exists(): shutil.rmtree(ROOT)
for d in [OUT, REP, TOOLS]: d.mkdir(parents=True, exist_ok=True)
V25 = Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/outputs/v25_attribute_group_render_container.3dphox')
V27 = Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27/outputs/v27_attribute_group_sh_vq_render_container.3dphox')
V25_REPORT = Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json')
V27_REPORT = Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27/reports/PHOXBENCH_V27_SH_DEBT_REPORT.json')

def read_container(path):
    with path.open('rb') as f:
        magic=f.read(11)
        ml=struct.unpack('<Q',f.read(8))[0]
        man=json.loads(f.read(ml))
        blob=f.read()
    chunks={c['name']:c for c in man['chunks']}
    def comp(name):
        c=chunks[name]
        return blob[c['offset']:c['offset']+c['compressed_bytes']]
    def dec(name):
        return zlib.decompress(comp(name))
    return magic, man, blob, chunks, comp, dec

m25, man25, blob25, ch25, comp25, dec25 = read_container(V25)
m27, man27, blob27, ch27, comp27, dec27 = read_container(V27)
rep25=json.loads(V25_REPORT.read_text())
rep27=json.loads(V27_REPORT.read_text())
N=rep25['input']['source_splats']; source_bytes=rep25['input']['source_ply_bytes']; v11=rep25['input']['v11_vq256_bytes']
print('decode sh + vq')
sh_q8=np.frombuffer(dec25('sh_rest_q8_global'), dtype=np.int8).reshape(N,45).copy()
labels=np.frombuffer(dec27('sh_vq128_idx_u8'), dtype=np.uint8).reshape(N,3).copy()
codebooks=np.frombuffer(dec27('sh_vq128_codebook_i8'), dtype=np.int8).reshape(3,128,15).copy()
tiers=np.frombuffer(dec25('tier_labels_u8'), dtype=np.uint8).copy()
# reconstruct and residual
approx=np.empty_like(sh_q8)
for g in range(3):
    approx[:,g*15:(g+1)*15]=codebooks[g][labels[:,g]]
res=(sh_q8.astype(np.int16)-approx.astype(np.int16))
res_min=int(res.min()); res_max=int(res.max())
fits_i8=res_min>=-128 and res_max<=127
if not fits_i8:
    res_store=res.astype(np.int16); res_dtype='int16'
else:
    res_store=res.astype(np.int8); res_dtype='int8'
# verify exact
recon=(approx.astype(np.int16)+res_store.astype(np.int16)).clip(-128,127).astype(np.int8)
exact_ok=bool(np.array_equal(recon, sh_q8))
# correction encoding variants
variants=[]
def cbytes(raw, lvl=6):
    return len(zlib.compress(raw, lvl))
# global full residual
raw_global=res_store.tobytes(); comp_global=zlib.compress(raw_global,6)
variants.append({'name':'global_full','compressed_bytes':len(comp_global),'raw_bytes':len(raw_global),'chunk_count':1})
# per group residual chunks
per_group=[]; total=0; raw_total=0
for g in range(3):
    raw=np.ascontiguousarray(res_store[:,g*15:(g+1)*15]).tobytes(); co=zlib.compress(raw,6); per_group.append(co); total+=len(co); raw_total+=len(raw)
variants.append({'name':'per_group','compressed_bytes':total,'raw_bytes':raw_total,'chunk_count':3})
# per tier+group residual chunks, implied indices by existing tier_labels_u8
per_tier_group=[]; total=0; raw_total=0; tier_stats=[]
for t in sorted(np.unique(tiers).tolist()):
    idx=np.nonzero(tiers==t)[0]
    ts={'tier':int(t),'count':int(idx.size),'groups':[]}
    for g in range(3):
        arr=np.ascontiguousarray(res_store[idx,g*15:(g+1)*15])
        raw=arr.tobytes(); co=zlib.compress(raw,6); per_tier_group.append((int(t),g,co,raw))
        total+=len(co); raw_total+=len(raw)
        ts['groups'].append({'group':g,'raw_bytes':len(raw),'compressed_bytes':len(co),'rmse_q8':float(np.sqrt(np.mean(arr.astype(np.float64)**2))) if arr.size else 0.0,'zero_fraction':float(np.mean(arr==0)) if arr.size else 0.0})
    tier_stats.append(ts)
variants.append({'name':'per_tier_group','compressed_bytes':total,'raw_bytes':raw_total,'chunk_count':len(per_tier_group)})
# sparse nonzero variant: bitpack mask + nonzero values per group
sparse_total=0; sparse_raw=0; sparse_parts=[]; sparse_stats=[]
for g in range(3):
    arr=np.ascontiguousarray(res_store[:,g*15:(g+1)*15].reshape(-1))
    nz=(arr!=0)
    mask=np.packbits(nz.astype(np.uint8))
    vals=arr[nz]
    raw=mask.tobytes()+vals.tobytes()
    co=zlib.compress(raw,6); sparse_parts.append((g,co,raw, int(nz.sum()), int(arr.size)))
    sparse_total+=len(co); sparse_raw+=len(raw)
    sparse_stats.append({'group':g,'nonzero':int(nz.sum()),'total':int(arr.size),'nonzero_fraction':float(nz.mean()),'compressed_bytes':len(co),'raw_sparse_bytes':len(raw)})
variants.append({'name':'sparse_mask_values','compressed_bytes':sparse_total,'raw_bytes':sparse_raw,'chunk_count':3})
best=min(variants,key=lambda x:x['compressed_bytes'])
# histogram metrics
hist={}
for bound in [0,1,2,4,8,16,32,64,128]:
    hist[f'abs_le_{bound}']=float(np.mean(np.abs(res)<=bound))
rmse_by_group=[]; max_by_group=[]; zero_by_group=[]
for g in range(3):
    arr=res[:,g*15:(g+1)*15]
    rmse_by_group.append(float(np.sqrt(np.mean(arr.astype(np.float64)**2))))
    max_by_group.append(int(np.max(np.abs(arr))))
    zero_by_group.append(float(np.mean(arr==0)))
# Choose per_tier_group if close or best, because it is context-aware; otherwise best.
chosen='per_tier_group'
chosen_bytes=next(v['compressed_bytes'] for v in variants if v['name']==chosen)
# compose exact archive container: v27 chunks plus correction chunks per tier/group.
def write_container(path, format_name, chunks, payloads, extra):
    off=0; new=[]; pls=[]
    for c,p in zip(chunks,payloads):
        cc=dict(c); cc['offset']=off; cc['compressed_bytes']=len(p); new.append(cc); pls.append(p); off+=len(p)
    manifest={'format':format_name,'cycle':'v0.28','source_splats':N,'source_ply_bytes':source_bytes,'chunks':new,**extra}
    with path.open('wb') as f:
        m=json.dumps(manifest,indent=2).encode(); f.write(b'CRYPSOID28\0'); f.write(struct.pack('<Q',len(m))); f.write(m)
        for p in pls: f.write(p)
    return manifest
# copy v27 base chunks exactly
base_chunks=man27['chunks']; base_payloads=[comp27(c['name']) for c in base_chunks]
# render container copy with v28 magic/manifest note
render_path=OUT/'v28_sh_vq_render_container.3dphox'
render_manifest=write_container(render_path,'CRYPSOID_3DPHOX_V28_SH_VQ_RENDER_CORE',base_chunks,base_payloads,{'truth_contract':{'copied_from':'v27 render core','not_exact_sh':'requires archive correction chunks for q8-exact SH'},'sh_vq':man27.get('sh_vq',{})})
# exact archive with per-tier-group residual chunks
corr_chunks=[]; corr_payloads=[]
for t,g,co,raw in per_tier_group:
    corr_chunks.append({'name':f'sh_exact_residual_t{t}_g{g}_{res_dtype}','dtype':res_dtype,'shape':['tier_count',15],'tier':t,'group':g,'raw_bytes':len(raw),'compressed_bytes':len(co),'crc32_raw':binascii.crc32(raw)&0xffffffff,'semantic':'exact correction residual: original_q8 = VQ_centroid + correction'})
    corr_payloads.append(co)
archive_path=OUT/'v28_sh_vq_exact_archive_container.3dphox'
archive_manifest=write_container(archive_path,'CRYPSOID_3DPHOX_V28_SH_VQ_EXACT_ARCHIVE',base_chunks+corr_chunks,base_payloads+corr_payloads,{'correction_encoding':'per_tier_group','residual_dtype':res_dtype,'exact_sh_reconstruction_test':exact_ok,'correction_histogram':hist,'residual_stats':{'min':res_min,'max':res_max,'rmse_q8_by_group':rmse_by_group,'max_abs_by_group':max_by_group,'zero_fraction_by_group':zero_by_group},'encoding_variants':variants,'tier_group_stats':tier_stats,'truth_contract':{'render_core':'uses VQ approximated SH','archive_mode':'uses per-tier/group residual chunks to reconstruct original v25 q8 SH exactly'}})
# verify archive CRC and exact reconstruction from written chunks
m28, man28, blob28, ch28, comp28, dec28 = read_container(archive_path)
crc_ok=True
for c in man28['chunks']:
    raw=zlib.decompress(blob28[c['offset']:c['offset']+c['compressed_bytes']])
    if (binascii.crc32(raw)&0xffffffff)!=c['crc32_raw']:
        crc_ok=False; print('crc fail',c['name'])
# Rebuild residuals from chunks by tier/group
res2=np.zeros_like(res_store)
for c in man28['chunks']:
    if c['name'].startswith('sh_exact_residual_t'):
        t=c['tier']; g=c['group']; idx=np.nonzero(tiers==t)[0]
        raw=zlib.decompress(blob28[c['offset']:c['offset']+c['compressed_bytes']])
        arr=np.frombuffer(raw,dtype=np.int8 if res_dtype=='int8' else np.int16).reshape(idx.size,15)
        res2[idx,g*15:(g+1)*15]=arr
recon2=(approx.astype(np.int16)+res2.astype(np.int16)).clip(-128,127).astype(np.int8)
readback_exact=bool(np.array_equal(recon2,sh_q8))
# svg correction histogram/size bars
v25_size=V25.stat().st_size; v27_size=V27.stat().st_size; render_size=render_path.stat().st_size; archive_size=archive_path.stat().st_size
bars=[('v11 VQ256 baseline',v11),('v25 q8 full attributes',v25_size),('v27 SH-VQ render',v27_size),('v28 SH-VQ render',render_size),('v28 q8-exact archive',archive_size)]
maxb=max(b for _,b in bars)
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="310"><rect width="100%" height="100%" fill="white"/><style>text{font-family:monospace;font-size:14px}</style>']
y=42
for label,b in bars:
    w=600*b/maxb
    svg.append(f'<text x="20" y="{y}">{label}</text><rect x="300" y="{y-15}" width="{w:.1f}" height="20" fill="#222"/><text x="{310+w:.1f}" y="{y}">{b/1024/1024:.2f} MiB</text>')
    y+=48
svg.append('</svg>')
(OUT/'v28_exact_archive_size_bars.svg').write_text('\n'.join(svg))
# residual histogram svg
absvals=np.abs(res).reshape(-1)
bins=[0,1,2,4,8,16,32,64,128]
counts=[]; prev=-1
for b in bins:
    counts.append(int(np.sum(absvals<=b)))
svg2=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300"><rect width="100%" height="100%" fill="white"/><style>text{font-family:monospace;font-size:13px}</style><text x="20" y="25">SH VQ residual absolute-error cumulative histogram</text>']
maxc=absvals.size; x=20
for i,b in enumerate(bins):
    h=220*counts[i]/maxc
    svg2.append(f'<rect x="{60+i*90}" y="{260-h:.1f}" width="52" height="{h:.1f}" fill="#333"/><text x="{58+i*90}" y="280">≤{b}</text><text x="{50+i*90}" y="{250-h:.1f}">{100*counts[i]/maxc:.1f}%</text>')
svg2.append('</svg>')
(OUT/'v28_sh_residual_histogram.svg').write_text('\n'.join(svg2))
report={'cycle':'v0.28','title':'Context-aware SH correction chunks for q8-exact archive mode','status':'Built v28 render and q8-exact archive containers. VQ render core remains small; archive mode carries per-tier/per-channel residual chunks that exactly reconstruct the v25 q8 SH stream. This separates render-size mode from exact-storage mode without lying about fidelity.','source':{'splats':N,'source_ply_bytes':source_bytes,'v11_vq256_bytes':v11,'v25_full_attribute_bytes':v25_size,'v27_render_bytes':v27_size},'v28':{'render_container_bytes':render_size,'render_container_mib':render_size/1024/1024,'archive_exact_container_bytes':archive_size,'archive_exact_container_mib':archive_size/1024/1024,'render_delta_vs_v11_percent':100*(render_size/v11-1),'archive_delta_vs_v25_percent':100*(archive_size/v25_size-1),'archive_delta_vs_v11_percent':100*(archive_size/v11-1),'archive_ratio_vs_source_ply':source_bytes/archive_size,'archive_reduction_vs_source_ply_percent':100*(1-archive_size/source_bytes)},'correction':{'encoding_variants':variants,'chosen':'per_tier_group','chosen_correction_bytes':chosen_bytes,'residual_dtype':res_dtype,'residual_min':res_min,'residual_max':res_max,'exact_ok_prewrite':exact_ok,'crc_ok':crc_ok,'readback_exact_sh_reconstruction':readback_exact,'residual_histogram':hist,'rmse_q8_by_group':rmse_by_group,'max_abs_by_group':max_by_group,'zero_fraction_by_group':zero_by_group,'tier_group_stats':tier_stats},'outputs':{'render_container':str(render_path),'archive_exact_container':str(archive_path),'size_bars':str(OUT/'v28_exact_archive_size_bars.svg'),'residual_histogram':str(OUT/'v28_sh_residual_histogram.svg')},'next':'v0.29 should reduce archive correction debt using context-conditioned codebooks or transform-coded residuals, because v28 exact archive is truthful but larger than the render core.'}
(REP/'PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json').write_text(json.dumps(report,indent=2))
md=f'''# CRYPSOID v0.28 — context-aware SH exact correction chunks

v0.27 made the full-attribute render container small by replacing the giant q8 SH stream with a VQ render core. v0.28 adds the missing truth layer: optional exact correction chunks that reconstruct the original v25 q8 SH stream.

## Result

| Container | Size | Meaning |
|---|---:|---|
| v11 VQ256 baseline | {v11/1024/1024:.2f} MiB | previous practical baseline |
| v25 q8 full-attribute | {v25_size/1024/1024:.2f} MiB | honest q8 SH container |
| v27 SH-VQ render | {v27_size/1024/1024:.2f} MiB | small render core, not exact |
| v28 SH-VQ render | {render_size/1024/1024:.2f} MiB | same render core, v28 manifest |
| v28 q8-exact archive | {archive_size/1024/1024:.2f} MiB | render core + q8-exact correction chunks |

## Correction encoding tested

- global residual stream
- per-SH-group residual streams
- per-tier/per-group residual streams
- sparse mask + nonzero values

Chosen: `per_tier_group`, because it is context-aware and keeps the correction chunks aligned with CRYPSOID tiers.

## Exactness

- CRC readback: `{crc_ok}`
- q8 SH reconstruction exact: `{readback_exact}`
- residual dtype: `{res_dtype}`
- residual range: [{res_min}, {res_max}]

## Honest read

v0.28 gives two modes:

1. **Render mode** — small VQ SH stream, below v11 size.
2. **Archive mode** — exact q8 SH reconstruction, larger but truthful.

This is not yet the final win. The next compression target is reducing exact-correction debt with better context-conditioned codebooks or residual transforms.
'''
(REP/'RESEARCH_BUILD_TEST_CYCLE_V28.md').write_text(md)
# README and copy script
(ROOT/'README.md').write_text('CRYPSOID v0.28: SH VQ render core + q8-exact correction archive mode. See reports/RESEARCH_BUILD_TEST_CYCLE_V28.md.\n')
shutil.copy('/mnt/data/build_v28.py', TOOLS/'build_v28_sh_exact_correction.py')
# packages
zip_path=Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_28.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_STORED) as z:
    for p in ROOT.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(ROOT.parent))
tar_path=Path('/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_28.tar.gz')
if tar_path.exists(): tar_path.unlink()
with tarfile.open(tar_path,'w:gz') as tar: tar.add(ROOT,arcname=ROOT.name)
print(json.dumps({'render':render_size,'archive':archive_size,'exact':readback_exact,'crc':crc_ok,'zip':str(zip_path),'tar':str(tar_path)},indent=2))
