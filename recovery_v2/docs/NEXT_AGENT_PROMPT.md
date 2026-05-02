# NEXT AGENT PROMPT — Continue CRYPSOID from recovered state

You are continuing the CRYPSOID `.3dphox` project after a chat-loss recovery. Do not restart from scratch.

Current verified continuation anchor:

```text
CRYPSOID v0.27
Format: CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V27_SH_VQ_RENDER
Container: /mnt/data/v27_attribute_group_sh_vq_render_container (1).3dphox
Verified bundle: /mnt/data/CRYPSOID_v27_verified_continuation_bundle.zip
Original source: /mnt/data/Audi A5 Sportback(1).zip
Source splats: 763,800
Container size: 18,796,089 bytes / 17.93 MiB
Reduction vs logical source PLY: 89.57%
```

Critical truth:

- v0.27 is the newest real usable render artifact.
- v0.29 in the recovered workspace is a residual-transform harness that ran synthetic smoke only, not a real Audi compression result.
- v0.28/v0.29 should be folded in as harness/planning work, not treated as proven final build output.
- The next real phase is v0.30 Render Truth Gate.

Mission:

1. Build the v0.30 render truth gate.
2. Use the same camera path for original Audi PLY and v27 `.3dphox`.
3. Produce contact sheets and metrics: MAE, MSE, PSNR, SSIM, count parity, tier counts, decode/render time.
4. Only after visual/metric gate exists, resume exact SH correction / v29 real residual transform sweep.

Hard rules:

- Do not claim compression wins unless the report states which attribute groups are carried: XYZ, DC/opacity, scale, quaternion, SH residuals, tier labels.
- Do not promote SARC/phoxoid replacement as render path unless it beats splat parity under visual metrics.
- Do not rely on CPU DC/opacity preview as final truth; label it as a sanity preview.
- Preserve splatpack/native exact path as master/fallback/parity path.
