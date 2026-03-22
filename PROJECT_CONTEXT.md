# RL-Based Convex Lens Polishing — Project Context

## 1. 项目目标

用强化学习 (RL) 模拟凸面镜的小球头抛光加工过程。
Agent 通过控制工具路径和工艺参数（压力、转速、驻留时间），在 2D 镜面上规划
抛光策略，同时优化三个目标：面形精度 (RMS)、表面粗糙度 (Ra)、加工时间。

---

## 2. 计算环境

- **集群**: SLURM 调度，使用 enroot 容器 (nvidia+pytorch+25.05-py3)
- **GPU**: 4 × NVIDIA GB200 (189GB 显存/张), CUDA 13.0
- **CPU**: 144 核
- **前端节点**: 只能编辑文件，不能运行 Python
- **后端计算节点执行命令的方式**:
  ```bash
  ssh "$(squeue --me -h -t R -o "%N" | head -n1)" \
    "pid=\$(enroot list --fancy | awk 'NR==2{print \$2; exit}'); \
     enroot exec \$pid <你的命令>"
  ```
- **已安装的 pip 包**: torch, gymnasium, stable-baselines3, openpyxl, pyyaml, tensorboard

---

## 3. 实验数据

文件: `副本数据汇总正式版.xlsx`，5 个 Sheet (P/V/C/B/new)，共 218 条有效记录。

每条记录包含:
- 输入: 抛光压力(N), 转速(r/min), 抛光液浓度(wt%), 球头直径(mm)
- 输出: 去除函数范围(mm), 去除函数深度(micron/s), 去除体积(micron³/s)

各 Sheet 的控制变量:
- P: 压力 5~50N (固定转速 300rpm, 浓度 5wt%)
- V: 转速 200~1300rpm (固定压力 10N, 浓度 5wt%)
- C: 浓度 1~10wt% (固定压力 30N, 转速 800rpm)
- B: 球头直径 30/50mm 对比
- new: 含进给速率和偏移角的补充实验

---

## 4. 代码结构

```
src/
├── models/
│   └── tif_model.py           # TIF (Tool Influence Function) 预测模型
│                               #   MLP: (pressure, speed, conc) -> (width_mm, depth_um/s)
│                               #   2D 高斯去除函数生成
├── env/
│   ├── lens_surface.py        # 2D 镜面模型
│   │                          #   目标面: 球面 h = R - sqrt(R² - r²)
│   │                          #   初始误差: 正值为主(多余材料), ~2-5μm
│   │                          #   64×64 网格, 50mm 直径
│   ├── roughness_model.py     # 粗糙度经验模型
│   │                          #   Ra 乘性衰减, quality_factor 基于参数偏离最优值
│   │                          #   初始 Ra=200nm, 范围 [5, 500]nm
│   └── polishing_env.py       # Gymnasium 环境 (核心)
│                               #   见下方"环境设计"
├── agent/
│   ├── policy.py              # 自定义 CNN+MLP 特征提取器
│   │                          #   CNN(2ch map) + MLP(scalar) -> 256-dim feature
│   └── train.py               # SAC 训练, 支持 SubprocVecEnv 多进程并行
├── utils/
│   └── data_loader.py         # xlsx 数据加载与清洗
├── configs/
│   └── default.yaml           # 所有超参数配置
├── scripts/
│   ├── fit_tif.py             # 拟合 TIF 模型并保存 checkpoint
│   ├── test_env.py            # 环境冒烟测试
│   ├── evaluate.py            # 评估 + 生成静态图表
│   └── animate.py             # 生成展示动画 (GIF/MP4)
├── checkpoints/
│   └── tif_model.pt           # 已训练好的 TIF 模型权重
└── runs/                      # 训练输出 (模型, 日志, 图表, 动画)
```

---

## 5. 环境设计 (polishing_env.py)

### 观测空间 (Observation)
```
Dict:
  "maps": (2, 64, 64) float32   — [error_map_normalized, roughness_map_normalized]
  "scalar": (3,) float32        — [tool_x/half, tool_y/half, elapsed_time/budget]
```
- error_map 归一化: err / initial_rms, clip [-5, 5]
- roughness_map 归一化: rough / 200nm, clip [0, 2]

### 动作空间 (Action)
```
Box(-1, 1, shape=(5,)):
  [dx, dy, pressure, speed, dwell_time]
映射:
  dx, dy → ±7.5mm (lens_diameter × 0.15)
  pressure → [5, 50] N
  speed → [200, 1300] rpm
  dwell_time → [0.1, 5.0] s
  concentration 固定 5wt%
```

### 奖励函数 (Reward)
```
reward = 5.0 * rms_improvement_fraction
       + 1.0 * roughness_improvement_fraction
       - 0.01 * (dwell / time_budget)
       - 2.0 * overpolish_penalty          ← 关键: 惩罚误差变负(过度抛光)
       + 10.0 * (达标 bonus, if RMS < 0.1μm)

clip to [-5, 5]
```

overpolish_penalty = mean(min(error, 0)²) / initial_rms²

### 终止条件
- terminated: RMS < target_rms (0.1μm)
- truncated: step >= 200 或 elapsed_time >= 600s

---

## 6. TIF 模型

- 架构: MLP (3→64→64→2), 输入/输出均做标准化
- 训练数据: 218 条, 从 xlsx 清洗
- 精度: width MAE=4.9%, depth MAE=21% (depth 方差大因 new Sheet 有异常值)
- 2D 去除函数: 高斯 `depth * exp(-r²/(2σ²))`, 其中 σ = width / (2√(2ln2))
- checkpoint: `src/checkpoints/tif_model.pt`

---

## 7. RL 训练

- 算法: SAC (Stable-Baselines3)
- 策略网络: MultiInputPolicy + 自定义 CNN+MLP 特征提取器
  - CNN: Conv2d(2→32, stride=2) → Conv2d(32→64, stride=2) → flatten
  - concat with scalar(3-dim) → MLP(256→256) → action(5-dim)
- 并行环境: 64 个 SubprocVecEnv (利用 144 CPU 核)
- 训练速度: ~1787 fps, 1M 步约 9.5 分钟
- 关键超参: lr=1e-4, batch=256, buffer=100K, learning_starts=2000,
  train_freq=4, gradient_steps=4

### 训练命令
```bash
ssh "$(squeue --me -h -t R -o "%N" | head -n1)" \
  "pid=\$(enroot list --fancy | awk 'NR==2{print \$2; exit}'); \
   enroot exec \$pid python3 src/agent/train.py \
     --total-timesteps 1000000 --n-envs 64 --device cuda"
```

### 评估 & 动画命令
```bash
# 评估 (生成 6 张对比图)
enroot exec $pid python3 src/scripts/evaluate.py \
  --model src/runs/checkpoints/best_model.zip --seed 7

# 动画 (生成 GIF)
enroot exec $pid python3 src/scripts/animate.py \
  --model src/runs/checkpoints/best_model.zip \
  --output src/runs/polishing_demo.gif --seed 7
```

---

## 8. 当前训练结果

- Best eval reward: +0.50 (在 ~878K 步)
- 面形: RMS 2.12 → 1.90 μm (改善 ~10%)
- 粗糙度: Ra 200 → 177 nm
- Agent 学到: 中等压力(~23N), 中等转速(~654rpm), 选择性抛光高误差区域
- Agent 缺陷: 路径覆盖不够均匀, 倾向于只抛一个区域

---

## 9. 已知问题 & 待改进方向

### 已解决的问题
1. 粗糙度数值爆炸 → 改用 np.where + 双向 clip [5, 500]nm
2. critic_loss 爆炸 → 观测空间 clip + learning_starts 预热
3. 过度抛光 → 添加 overpolish_penalty (负误差平方惩罚)

### 待改进
1. **路径覆盖**: Agent 只在一个区域反复抛光, 需要添加覆盖率奖励或
   exploration bonus, 鼓励均匀扫描整个镜面
2. **训练量**: 1M 步时仍在改善, 可尝试 5M-10M 步
3. **Curriculum learning**: 先用简单误差面 (小振幅、低频) 训练,
   逐步增加误差复杂度
4. **Grid 分辨率**: 当前 64×64, 可提升到 128×128 获得更精细面形
5. **浓度可控**: 目前浓度固定 5wt%, 可加入动作空间
6. **多目标 Pareto**: 替换简单加权为 MORL 方法
7. **和真实数据对比**: 目前初始误差是数学生成的,
   有真实测量面形数据后可直接替换

---

## 10. 快速上手

```bash
# 1. 拟合 TIF 模型 (如果 checkpoint 不存在)
enroot exec $pid python3 src/scripts/fit_tif.py

# 2. 测试环境
enroot exec $pid python3 src/scripts/test_env.py

# 3. 训练
enroot exec $pid python3 src/agent/train.py \
  --total-timesteps 1000000 --n-envs 64 --device cuda

# 4. 评估
enroot exec $pid python3 src/scripts/evaluate.py \
  --model src/runs/checkpoints/best_model.zip

# 5. 生成动画
enroot exec $pid python3 src/scripts/animate.py \
  --model src/runs/checkpoints/best_model.zip \
  --output src/runs/polishing_demo.gif

# 所有超参数在 src/configs/default.yaml 中配置
```
