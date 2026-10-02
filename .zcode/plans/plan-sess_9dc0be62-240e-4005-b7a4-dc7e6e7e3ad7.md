# 根因分析与修复: 为什么会出错以至于需要校验

## 已完成的分析(plan模式下的静态审查)

**为什么会出错**: 闭环编码器在罕见内容上产生"重建≠解码"的比特流——编码器自以为写了X,libwebp解码器读出Y,从MB(41,36)起整个流错位(结构保留+~+35亮度偏移,fypdm图31.03dB vs Pillow 42.86dB)。解码校验是唯一能捕捉这类比特流级错误的层; GPU重建门(量化误差域47dB干净)对此失明。这正是"需要校验"的根本原因。

**已逐行排除**(对照vendor/libwebp-1.5.0源码):
- emit_block token树/EOB/零游程/上下文更新 — 对GetCoeffsFast逐位一致 ✓
- cat1-cat6幅值树与附加位 — 对GetLargeValue一致 ✓
- i4模式树(put_i4) — 对kYModesIntra4权威树逐节点一致 ✓
- i16/uv模式树与上下文 — 对ParseIntraMode一致 ✓
- nz上下文(GPU闭环对i16恒零化pos0,quantize(...,1)) ✓
- Y2仅DC时的简化重建规则(已镜像) ✓
- BANDS/CAT3-6小表 ✓

**头号嫌疑**: `gpuwebp/vp8_tables.py`的**KF_BMODE_PROBA(900项)/COEFFS_PROBA0(1056项)**大表单条目错误(生成`entropy_tables.inc`→entropy.dll,且Python/cupy路径共用同源表→两路径字节一致,与观测吻合)。单个错值→只有命中该概率条目的罕见模式组合才错位→"动漫没事/粗颗粒照片坏",坏MB恰用罕见的mode-7上下文。

## 执行步骤

1. **表全量比对**(最可能一击命中): 脚本比对vp8_tables.py两张大表 vs vendor/libwebp(tree_dec.c的kBModesProba + 默认系数概率表);发现错值→修正→重新生成entropy_tables.inc→重建entropy.dll
2. **(若表干净)bool writer验证**: 从ops日志用参考算术模拟重新编码,比对实际输出字节,定位进位/0xff-run路径问题
3. **(若仍干净)修好流解析器**(range减一约定+24位装载逐行镜像C实现),roundtrip验证后解析fypdm码流找首个发散token
4. **验证阶梯**:
   - fypdm: PSNR应从31.03→~44(重建47减色度损失)
   - 176张新基准(受影响图的输出会变——它们本来就是坏的)
   - 600×2确定性 + padding/精确A/B不变性必须保持
   - 全量完整校验: 兜底应从12→~7(只剩奇数尺寸),速度持平
   - 重建exe + 一次全量验证
   - git提交(附根因说明)