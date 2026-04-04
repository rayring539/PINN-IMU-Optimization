# IMU温度漂移物理模型框架 —— 面向PINN建模

## 1. 总体架构：三层物理因果链

IMU温度漂移的物理机制可分解为三层级联的PDE/ODE系统，每一层的输出作为下一层的输入：

```
环境温度变化 T_env(t)
       ↓
[Layer 1] 热传导方程 → 内部温度场 T(x, t)
       ↓
[Layer 2] 热弹性耦合方程 → 热应力 σ(x,T)、热变形 u(x,T)
       ↓
[Layer 3] 传感器误差模型 → 偏置漂移 B(T)、标度因子漂移 SF(T)
       ↓
传感器输出误差 δω(t), δa(t)
```

在PINN框架中，这三层方程分别作为物理损失项嵌入总损失函数：

```
L_total = L_data + λ₁·L_heat + λ₂·L_thermoelastic + λ₃·L_sensor
```

---

## 2. Layer 1：热传导方程（温度场演化）

### 2.1 控制方程

传感器内部温度场满足傅里叶热传导方程：

```
ρ·Cₚ·∂T/∂t = ∇·(k·∇T) + Q
```

其中：
- ρ : 材料密度 (kg/m³)
- Cₚ : 比热容 (J/(kg·K))
- k : 热导率 (W/(m·K))
- Q : 内部热源项 (W/m³)，包括振动耗散产热、电路发热等
- T = T(x, t) : 温度场

**一维简化形式**（适用于沿传感器厚度方向的温度梯度建模）：

```
ρ·Cₚ·∂T/∂t = k·∂²T/∂x² + Q
```

### 2.2 边界条件

| 边界类型 | 数学表达 | 物理含义 |
|---------|---------|---------|
| 第一类 (Dirichlet) | T|_∂Ω = T_env(t) | 外壳温度跟踪环境 |
| 第二类 (Neumann) | -k·∂T/∂n|_∂Ω = q | 给定热流密度 |
| 第三类 (Robin) | -k·∂T/∂n|_∂Ω = h·(T - T_env) | 对流换热 |

对于真空封装的HRG/高精度MEMS，内部热交换以辐射为主：

```
q_rad = ε·σ_SB·(T⁴ - T_env⁴)
```

其中 ε 为发射率，σ_SB 为Stefan-Boltzmann常数。

### 2.3 硅材料热物性的温度依赖

```
k_Si(T) = k₀·(T₀/T)^α     (α ≈ 1.3, 对于单晶硅)
Cₚ_Si(T) = a + b·T + c·T²   (多项式拟合)
```

### 2.4 PINN损失项

```
L_heat = (1/N_c) · Σ |ρCₚ·∂T̂/∂t - k·∂²T̂/∂x² - Q|²
       + (1/N_b) · Σ |BC_residual|²
```

其中 T̂ 为神经网络预测的温度场，N_c 为配点数，N_b 为边界点数。

---

## 3. Layer 2：热弹性耦合方程（温度→应力→变形）

### 3.1 热弹性耦合FEM方程

根据Nature Microsystems & Nanoengineering (2023)，完整的热弹性耦合方程为：

```
力学方程:    M·ü + K_u·u = K_ut·ΔT + f
热方程:      C_t·ΔṪ + K_t·ΔT = K_ut^T·u̇
```

其中：
- M : 质量矩阵
- K_u : 刚度矩阵
- K_ut : 热弹性耦合矩阵
- C_t : 比热矩阵
- K_t : 热导矩阵
- u : 位移向量
- ΔT : 温度变化向量
- f : 外力向量

### 3.2 热应力的连续体表达

在线弹性热应力理论下，应力-应变关系为：

```
σ_ij = C_ijkl · (ε_kl - α·ΔT·δ_kl)
```

其中：
- C_ijkl : 弹性常数张量
- ε_kl : 总应变
- α : 热膨胀系数 (CTE)
- ΔT = T - T_ref : 相对参考温度的温度变化
- δ_kl : Kronecker delta

**一维简化**：

```
σ = E(T) · (ε - α·ΔT)
```

### 3.3 杨氏模量的温度依赖（核心参数）

单晶硅杨氏模量随温度线性下降：

```
E_Si(T) = E₀ · (1 - k_E · ΔT)
```

其中：
- E₀ ≈ 170 GPa (室温下，取决于晶向)
- k_E ≈ 60 ppm/°C (杨氏模量温度系数, TCE)
- ΔT = T - T₀ (T₀ 通常取25°C)

石英（用于HRG）：

```
E_quartz(T) = E₀ · (1 - k_E · ΔT)
```

其中 k_E ≈ 80 ppm/°C。

### 3.4 热膨胀系数的温度依赖

```
α_Si(T) = α₀ + α₁·T + α₂·T²
```

注意：硅在某些低温条件下热膨胀系数可变为负值，这会导致品质因子发生数量级变化。

### 3.5 热应力引起的刚度变化

弹簧/梁的等效刚度（折叠梁模型）：

```
K(T) = E(T) · I(T) / L(T)³ · C_geom
```

其中：

```
I(T) = w(T)³ · h(T) / 12              (截面惯性矩)
w(T) = w₀ · (1 + α · ΔT)              (梁宽的热膨胀)
h(T) = h₀ · (1 + α · ΔT)              (梁高的热膨胀)
L(T) = L₀ · (1 + α · ΔT)              (梁长的热膨胀)
C_geom = 常数 (取决于振动模态)
```

展开至一阶近似：

```
K(T) ≈ K₀ · [1 + (TCE_eff)·ΔT]
TCE_eff = -k_E + (3-3+1)·α = -k_E + α
```

由于 |k_E| >> |α|，刚度主要随温度升高而下降。

### 3.6 谐振频率的温度依赖

```
f_res(T) = (1/2π) · √(K(T) / M_eff(T))
```

一阶近似：

```
f_res(T) ≈ f₀ · [1 + TCf · ΔT]
```

其中温度频率系数：

```
TCf ≈ (1/2) · (TCE_eff - TCM_eff)
    ≈ -(k_E/2)    (简化，忽略质量变化)
    ≈ -30 ppm/°C   (对于硅)
```

### 3.7 封装引起的热-机械应力

多层封装结构（硅芯片-粘接胶-陶瓷基板）的CTE失配应力：

```
σ_pkg = E_Si · (α_sub - α_Si) · ΔT / (1 + E_Si·t_Si / (E_sub·t_sub))
```

此应力叠加在谐振梁上，等效为附加轴向力 F_thermal：

```
F_thermal = σ_pkg · A_beam
```

修正后的谐振频率：

```
f_res(T) = (1/2π) · √[(K₀ + B·F_thermal) / M_eff]
```

### 3.8 PINN损失项

```
L_thermoelastic = (1/N) · Σ |σ̂ - E(T)·(ε̂ - α·ΔT)|²
                + (1/N) · Σ |∇·σ̂ + ρ·f_body|²     (平衡方程)
                + (1/N) · Σ |ε̂ - (1/2)(∇û + ∇û^T)|²  (几何方程)
```

---

## 4. Layer 3-A：加速度计温度漂移物理模型（重点）

加速度计是IMU中温度漂移最严重的传感器之一。根据检测原理不同，分为**电容式**和**谐振式**两大类，物理漂移机制有本质区别。

---

### 4.1 电容式加速度计 —— 基本工作原理

电容式加速度计通过检测惯性力引起的质量块位移来测量加速度。核心结构为弹簧-质量块-电容板系统：

```
牛顿第二定律:  m·ẍ + c·ẋ + K·x = m·a_input
静态平衡:      x_static = m·a / K
电容检测:      ΔC = C₁ - C₂ = ε₀·A·[1/(d₀-x) - 1/(d₀+x)]
             ≈ 2·ε₀·A·x / d₀²    (x << d₀时)
```

其中：
- m : 质量块质量
- K : 弹簧刚度
- d₀ : 标称电容间隙
- A : 电极面积
- ε₀ : 真空介电常数

**标度因子** (开环模式)：

```
SF_open = ΔC / a = 2·ε₀·A·m / (K·d₀²)
```

温度通过改变 K、d₀、A、m 影响标度因子和偏置。

---

### 4.2 电容式加速度计 —— 偏置温度漂移 (TDB) 的物理推导

#### 4.2.1 热变形引起的差分间隙变化

**核心物理机制**：封装基板(陶瓷/玻璃)与硅结构的CTE失配导致锚点位移，使质量块相对固定电极产生非对称偏移。

**多层封装热变形模型**（He et al., Sens. Act. A, 2016; PMC Sensors 2019）：

五层结构：陶瓷壳体 → 粘接胶 → 硅基底(handle wafer) → SiO₂绝缘层 → 硅器件层

等效膨胀比（加权平均法）：

```
α_eq = Σᵢ (αᵢ · Eᵢ · tᵢ) / Σᵢ (Eᵢ · tᵢ)
```

其中 αᵢ, Eᵢ, tᵢ 分别为各层的CTE、杨氏模量、厚度。

**锚点位移模型**：

```
Δx_anchor(T) = (α_eq - α_Si) · L_anchor · ΔT
```

其中 L_anchor 为锚点到器件中心的距离。

**质量块刚性位移**（由不对称锚点热变形引起）：

对于可动电极锚点在 x₁ 和 x₂ 处的情况：

```
Δx_mass = (α_eq - α_Si) · (x₁ + x₂) / 2 · ΔT
```

**差分间隙变化**：

```
固定电极侧:  Δd_fixed = (α_eq - α_Si) · L_fixed · ΔT
可动电极侧:  Δd_movable = (α_eq - α_Si) · L_movable · ΔT

差分间隙变化:
Δd = Δd_movable - Δd_fixed = (α_eq - α_Si) · (L_movable - L_fixed) · ΔT
```

### 7.2 TDB解析公式

**开环模式**：

```
TDB_open = (1/g) · (ε₀·A·V_bias²) / (K·d₀³) · ∂(Δd)/∂T
         = (1/g) · (ε₀·A·V_bias²) / (K·d₀³) · (α_eq - α_Si) · ΔL
```

其中 ΔL = L_movable - L_fixed 表征结构不对称性。

**简化物理公式**（当无静电力时，纯热变形引起的偏置漂移）：

```
TDB = ∂B/∂T = g · Δd(T) / d₀ · (K_left - K_right) / (K_left + K_right)
```

即 TDB 正比于：(1) CTE失配 (α_eq - α_Si)，(2) 锚点不对称性 ΔL，(3) 弹簧刚度不对称性。

**闭环模式** (Zhou et al., Sens. Act. A, 2019)：

```
TCB_closed = TCB_open    (闭环不改善偏置温度系数)
```

即闭环模式下的偏置温度漂移与开环模式相同，因为TCB由质量块刚性位移决定，与力平衡方式无关。

### 7.3 考虑弹簧刚度不对称的完整TDB模型

实际制造中弹簧存在工艺偏差，左右刚度不完全相等：

```
K_left(T) = E(T) · I_left(T) / L_left(T)³ · C_geom
K_right(T) = E(T) · I_right(T) / L_right(T)³ · C_geom
```

由于DRIE（深反应离子刻蚀）工艺偏差：

```
w_left = w₀ + δw,    w_right = w₀ - δw    (对称偏差)
```

则刚度差异：

```
ΔK/K₀ = 3·δw/w₀    (一阶近似)
```

此刚度差异在温度变化时会因杨氏模量和几何尺寸变化而改变，产生额外的TDB：

```
TDB_stiffness = m·g·(ΔK/K₀)·(∂E/∂T)/E₀
              = m·g·(3δw/w₀)·(-k_E)
```

---

### 4.3 电容式加速度计 —— 标度因子温度漂移 (TDSF) 的物理推导

#### 4.3.1 开环模式 TDSF

标度因子（灵敏度）：

```
SF_open = ΔC/a = 2·ε₀·A(T)·m / [K(T)·d₀(T)²]
```

温度微分：

```
(1/SF)·∂SF/∂T = (1/A)·∂A/∂T - (1/K)·∂K/∂T - (2/d₀)·∂d₀/∂T
```

代入各项的温度依赖：

```
(1/A)·∂A/∂T = 2·α_Si                        (面积: 二维膨胀)
(1/K)·∂K/∂T = -k_E + α_Si                   (刚度: 杨氏模量 + 几何)
(2/d₀)·∂d₀/∂T = 2·(α_eq - α_Si)·L/d₀       (间隙: CTE失配)
```

因此：

```
TDSF_open = 2α_Si - (-k_E + α_Si) - 2(α_eq - α_Si)·L/d₀
          = k_E + α_Si - 2(α_eq - α_Si)·L/d₀
```

**两个竞争项**：
- 第一项 (k_E + α_Si ≈ +62.6 ppm/°C)：正号，由杨氏模量软化引起
- 第二项 (-2(α_eq-α_Si)·L/d₀)：负号，由CTE失配热变形引起

通常第二项绝对值更大，因此 TDSF_open 为负。

#### 4.3.2 闭环模式 TDSF（重要区别！）

闭环模式下，惯性力由静电反馈力平衡，弹簧力不参与：

```
m·a = F_electrostatic = ε₀·A(T)·V_fb² / [2·d₀(T)²]
```

因此：

```
SF_closed = V_fb / a = √[2·m·d₀(T)² / (ε₀·A(T))]
```

温度微分：

```
TCSF_closed = (2/d₀)·∂d₀/∂T - (1/A)·∂A/∂T
            = 2·α_eq_d - 2·α_Si
```

**关键结论：闭环模式的TCSF与弹簧刚度温度依赖无关！**

这意味着只要抑制热变形（软粘接、居中锚点等），闭环TCSF可以做得很小。

---

### 4.4 谐振式加速度计 —— 温度漂移物理模型

谐振式加速度计通过检测双端音叉(DETF)谐振频率的变化来测量加速度，是高精度应用的主流方案。

#### 4.4.1 基本力-频特性

无轴向力时，DETF谐振频率：

```
f₀ = (1/2π) · √(K_eff / M_eff)
```

其中有效刚度和有效质量：

```
K_eff = A_mode · E_Si · w³ · h / (12 · L³)
M_eff = ρ_Si · w · h · L · B_mode
```

A_mode, B_mode 为与振动模态相关的常数。

当轴向力 F 加载到谐振梁上时：

```
K_eff(F) = K_eff + B_mode · F / L
```

因此谐振频率随轴向力变化：

```
f(F) = f₀ · √(1 + B_mode · F / (L · K_eff))
     ≈ f₀ · (1 + B_mode · F / (2·L·K_eff))    (F << K_eff·L时)
```

差分输出（两个DETF受相反的力）：

```
Δf = f₁ - f₂ ≈ f₀ · B_mode · F / (L · K_eff) ∝ a_input
```

#### 4.4.2 温度引起的频率漂移 —— 三大物理来源

**来源1：杨氏模量的温度依赖（材料固有效应）**

```
f₀(T) = f₀(T₀) · √(E(T)/E(T₀))
       ≈ f₀(T₀) · (1 - k_E·ΔT/2)
```

对频率的温度系数贡献：

```
TCf_material = -k_E/2 ≈ -30 ppm/°C
```

**来源2：几何尺寸的热膨胀**

```
w(T) = w₀(1 + α·ΔT), h(T) = h₀(1 + α·ΔT), L(T) = L₀(1 + α·ΔT)
```

代入 f₀ 公式后：

```
TCf_geometry = α·(3/2 - 3/2 + 1/2 - 1/2) ≈ 0    (各项几乎抵消)
```

几何膨胀对频率的影响远小于杨氏模量效应。

**来源3：封装热-机械应力（最复杂、最难预测）**

四层封装模型（陶瓷壳→粘接胶→硅基底→硅器件层）中，CTE失配产生热应力，等效为谐振梁上的附加轴向力：

```
σ_thermal(T) = E_Si · (α_sub_eff - α_Si) · ΔT / (1 + E_Si·t_Si/(E_sub·t_sub))
F_thermal(T) = σ_thermal · A_beam = σ_thermal · w · h
```

频率修正：

```
f(T) = f₀(T) · √(1 + B_mode · F_thermal(T) / (L · K_eff(T)))
```

频率温度系数的封装贡献：

```
TCf_package ∝ E_Si · (α_sub_eff - α_Si) / K_eff
```

**关键结论**：TCf_package 的量级与 TCf_material 相当（约数十ppm/°C），甚至可能更大。这意味着封装设计对温度漂移的影响不可忽略。

#### 4.4.3 综合频率温度模型

```
f(T) = f₀ · √[E(T)/E₀] · √[(K_eff + B·F_thermal(T)) / K_eff]
     · [1 + geometric_correction_terms]
```

一阶展开：

```
TCf_total = TCf_material + TCf_geometry + TCf_package
          = -k_E/2 + ~0 + E_Si·(α_sub_eff - α_Si)·w·h / (2·L·K_eff)
```

#### 4.4.4 差分频率输出的温度漂移

理想差分消除共模温度漂移，但实际中两个谐振器的温度系数不完全相同：

```
f₁(T) = f₁₀ · [1 + TCf₁·ΔT]
f₂(T) = f₂₀ · [1 + TCf₂·ΔT]
```

差分输出：

```
Δf(T) = f₁(T) - f₂(T) = (f₁₀ - f₂₀) + (f₁₀·TCf₁ - f₂₀·TCf₂)·ΔT
```

零偏温度漂移：

```
TDB_resonant = f₁₀·TCf₁ - f₂₀·TCf₂
             = f₀·(TCf₁ - TCf₂) + (f₁₀ - f₂₀)·TCf_avg
```

其中第一项由工艺偏差引起，第二项由初始频差引起。

**温差比方法**（用于补偿）：

```
a_comp = (Δf - r·Σf) / SF
r = (TCf₁ - TCf₂) / (TCf₁ + TCf₂)    (温差比系数)
```

#### 4.4.5 PINN中的加速度计物理损失项

**电容式加速度计**：

```
L_acc_cap = λ₁·|B̂(T) - g·Δd̂(T)/d₀·(ΔK̂/K₀)|²        (TDB物理约束)
          + λ₂·|ŜF(T) - 2ε₀Â(T)m/(K̂(T)·d̂₀²)|²       (SF物理约束)
          + λ₃·|K̂(T) - E(T)·Î(T)/L̂(T)³·C_geom|²      (刚度-温度关系)
          + λ₄·|Δd̂(T) - (α_eq-α_Si)·L_anchor·ΔT|²     (热变形约束)
```

**谐振式加速度计**：

```
L_acc_res = λ₁·|f̂(T) - f₀·√(E(T)/E₀)·√(1+B·F̂_th/K̂)|²  (频率物理模型)
          + λ₂·|F̂_thermal - σ̂·w·h|²                       (热应力→轴力)
          + λ₃·|σ̂ - E_Si·(α_sub-α_Si)·ΔT/(1+η)|²         (CTE失配应力)
          + λ₄·|Δf̂ - r·Σf̂|² · (零加速度工况)             (差分温度一致性)
```

---

### 4.5 加速度计温度漂移中的动态效应

#### 4.5.1 温度变化率 (dT/dt) 效应

由于传感器内部温度场的非均匀性（热传导滞后），偏置不仅取决于当前温度，还取决于温度变化速率：

```
B(T, Ṫ) = B_static(T) + B_dynamic(Ṫ)
         = [β₀ + β₁·ΔT + β₂·ΔT²] + [γ₁·Ṫ + γ₂·Ṫ²]
```

物理解释：
- β项：准静态热变形、刚度变化
- γ项：内部温度梯度引起的非均匀热应力

**γ项的物理来源**：当温度快速变化时，锚点（与基板热接触好）温度响应快于质量块（悬浮，热阻大），产生瞬态温差：

```
ΔT_internal(t) = T_anchor(t) - T_mass(t) ∝ τ_thermal · dT_env/dt
```

其中 τ_thermal 为传感器内部热弛豫时间常数：

```
τ_thermal = ρ·Cₚ·L² / k    (特征热扩散时间)
```

#### 4.5.2 温度滞回效应

温升和温降过程中偏置轨迹不同（迟滞现象），物理来源：
- 粘接胶的粘弹性松弛（蠕变）
- 残余应力的热历史依赖
- 封装材料的非线性热膨胀

数学描述（Preisach模型简化形式）：

```
σ_residual(T, history) = σ₀ + ∫ H(T(t'), dT/dt') dt'
```

此项难以用简单ODE描述，但可在PINN中通过引入隐状态变量建模：

```
dh/dt = f_NN(h, T, Ṫ)    (h为隐状态，由神经网络学习)
B_hysteresis = g_NN(h)     (迟滞偏置由隐状态映射)
```

#### 4.5.3 完整的三因子温度漂移模型

综合考虑温度、温变率和温度乘积项（Applied Sciences, 2019）：

```
B(T, Ṫ, ΔT) = a₀ + a₁·T + a₂·T² + a₃·T³
             + b₁·Ṫ + b₂·Ṫ²
             + c₁·T·ΔT + c₂·Ṫ·ΔT
```

三因子矩阵：

```
C = [T, T², T³, Ṫ, Ṫ², T·ΔT, Ṫ·ΔT]
B = C · θ    (θ为参数向量)
```

---

### 4.6 电路部分的温度漂移

温度不仅影响机械结构，还影响读出电路：

```
V_out(T) = G(T) · [ΔC(T)/C_ref(T)] + V_offset(T)
```

其中：
- G(T) : 放大器增益温度漂移（典型值: 数十 ppm/°C）
- C_ref(T) : 参考电容温度漂移
- V_offset(T) : 运放失调电压温度漂移（典型值: 数 μV/°C）

对于力平衡式加速度计，永磁体的磁感应强度也随温度变化：

```
B_magnet(T) = B₀ · [1 - α_B · (T - T₀)]
```

α_B 为永磁体温度系数（NdFeB: ~-1200 ppm/°C, SmCo: ~-300 ppm/°C）。

这直接影响力平衡电流→加速度的转换关系：

```
a = (B_magnet(T) · I · L_coil) / m
SF_force = B_magnet(T) · L_coil / m ∝ (1 - α_B · ΔT)
```

---

### 4.7 加速度计关键参考文献

| # | 论文 | 年份 | 核心贡献 |
|---|------|------|---------|
| 1 | He et al., "Analytical study and compensation for temperature drifts of bulk silicon MEMS capacitive accelerometer", Sens. Act. A | 2016 | TDB/TDSF解析模型，热变形与刚度分离 |
| 2 | Zhou et al., "Analytical study of TCB and TCSF in closed-loop mode", Sens. Act. A | 2019 | 闭环模式TCB=开环，TCSF与刚度无关 |
| 3 | Zhang et al., "Analytical study and thermal compensation for MEMS accelerometer with anti-spring structure", JMEMS | 2020 | GAS结构解析热漂模型，COMSOL验证 |
| 4 | PMC, "Analysis of Thermally Induced Packaging Effects on Frequency Drift of MEMS Resonant Accelerometer" | 2023 | E(T)公式，四层封装热应力模型 |
| 5 | Jiang et al., "Analysis of Frequency Drift of Silicon MEMS Resonator with Temperature", Micromachines | 2021 | 四层封装CTE失配应力→频率漂移 |
| 6 | PMC, "Thermal Drift Investigation of SOI-Based MEMS Capacitive Sensor with Asymmetric Structure" | 2019 | 非对称梳齿结构TDB/TDSF公式 |
| 7 | Zhang et al., "Temperature bias drift phase-based compensation for MEMS accelerometer", Nanomanuf. Metrol. | 2023 | 偏置漂移由敏感刚度主导的物理证据 |
| 8 | Springer, "Analysis of Temperature Stability and Change of Resonant Frequency", IJPEM | 2022 | 杨氏模量是谐振频率漂移主因的实验验证 |
| 9 | PMC, "Lightweight Thermal Compensation for MEMS Capacitive Accelerometer" | 2021 | TDB达1.3 mg/°C的实测，与梁刚度差成正比 |
| 10 | ScienceDirect, "Compensation of temperature effects in force-balanced MEMS accelerometers" | 2024 | 力平衡式加速度计三因素温度效应 |

---

## 5. Layer 3-B：陀螺仪温度漂移物理模型

### 7.1 基本振动方程

MEMS Coriolis振动陀螺仪的驱动/检测模态耦合方程：

```
驱动模态: m·ẍ + c_x·ẋ + k_x·x = F_drive
检测模态: m·ÿ + c_y·ẏ + k_y·y = -2m·Ω_z·ẋ + F_quadrature
```

其中 Ω_z 为输入角速率。

### 7.2 陀螺偏置的物理表达

根据Prikhodko et al. (2013) 和 MDPI Sensors (2020)，零速率偏置：

```
B_gyro = k_ag · Δ(1/τ) · sin(2θ_τ)
```

其中：
- k_ag : 角增益因子（取决于几何，温度稳定）
- Δ(1/τ) = 1/τ₁ - 1/τ₂ : 主轴阻尼差异
- τ₁, τ₂ : 两个主轴的阻尼时间常数
- θ_τ : 主阻尼轴方位角

偏置上界：

```
|B_gyro| ≤ k_ag · |Δ(1/τ)|
```

### 7.3 阻尼的温度依赖

总阻尼包含多种物理机制：

```
1/τ_total = 1/τ_air + 1/τ_TED + 1/τ_anchor + 1/τ_surface
```

**空气阻尼**（大气封装）：

```
1/τ_air(T) ∝ P / √(T)    (稀薄气体区域)
或
1/τ_air(T) ∝ μ(T) / d²   (连续流区域)
```

气体粘度温度依赖（Sutherland公式）：

```
μ(T) = μ₀ · (T/T₀)^(3/2) · (T₀ + S)/(T + S)
```

S 为Sutherland常数。

**热弹性阻尼 (TED)**：

Zener模型：

```
1/Q_TED = (E·α²·T₀) / (ρ·Cₚ) · (ω·τ_thermal) / (1 + (ω·τ_thermal)²)
```

其中热弛豫时间：

```
τ_thermal = w² / (π²·D_thermal)
D_thermal = k / (ρ·Cₚ)    (热扩散系数)
```

Lifshitz-Roukes模型（更精确，考虑热扩散长度）：

```
1/Q_TED = (E·α²·T₀) / (ρ·Cₚ) · [6/ζ² - 6/ζ³ · (sinh(ζ)+sin(ζ))/(cosh(ζ)+cos(ζ))]
```

其中 ζ = w·√(ω/(2·D_thermal))。

### 6.4 品质因子 (Q) 的温度模型

```
Q(T) = ω · τ_total(T) / 2
```

由于 E(T)、α(T)、k(T)、Cₚ(T) 均为温度函数，Q 表现为温度的复杂非线性函数。

### 5.5 陀螺偏置温度漂移

```
∂B_gyro/∂T = k_ag · ∂[Δ(1/τ)]/∂T · sin(2θ_τ)
            + k_ag · Δ(1/τ) · cos(2θ_τ) · 2·∂θ_τ/∂T
```

即偏置温度灵敏度同时取决于阻尼差异的温度变化和主阻尼轴方位角的温度变化。

### 5.6 标度因子的温度模型

```
SF_gyro(T) = (x_drive · ω_drive) / (ω_y² - ω_x²) · 2Ω / Q_sense(T)
```

简化的温度依赖：

```
∂SF/∂T ∝ ∂(Δω²)/∂T = ∂(ω_y² - ω_x²)/∂T
```

由于 ω ∝ √(K/M) ∝ √(E(T))：

```
∂ω²/∂T = -k_E · ω₀² + 封装应力项
```

### 5.7 PINN损失项

```
L_sensor = (1/N_d) · Σ |ŷ_sensor - y_measured|²    (数据项)
         + λ_bias · |B̂(T) - f_physics(K̂(T), d̂(T))|²  (偏置物理约束)
         + λ_sf · |ŜF(T) - g_physics(Ê(T), α̂(T))|²   (标度因子物理约束)
```

---

## 6. HRG半球谐振陀螺仪的热弹性动力学方程

### 7.1 薄壳热弹性方程

基于Kirchhoff-Love壳理论，半球壳谐振器的热弹性动力学方程：

```
ρ·h·∂²w/∂t² + D·∇⁴w + N_T·∇²w = 0
```

其中：
- w : 壳体法向位移
- D = E·h³/[12(1-ν²)] : 弯曲刚度
- N_T = E·α·ΔT·h/(1-ν) : 温度引起的膜力

### 7.2 温度与等效热载荷

```
M_T = ∫_{-h/2}^{h/2} [E·α·ΔT/(1-ν)] · z · dz    (热弯矩)
N_T = ∫_{-h/2}^{h/2} [E·α·ΔT/(1-ν)] · dz          (热膜力)
```

### 7.3 频率-温度关系

```
f(T) = f₀ · [1 + TCf·ΔT + TCf₂·ΔT² + ...]
```

HRG中 TCf ≈ -80 ppm/°C (石英)。

由于f(T)可实时测量，可反过来估算温度：

```
ΔT ≈ Δf / (f₀ · TCf)
```

### 6.4 频率-偏置漂移综合模型

```
B_HRG = a₀ + k₁·Δf + k₂·Δf² + k₃·Δf³
      + k₄·(dΔf/dt) + k₅·(dΔf/dt)²
      + k₆·Δf·(dΔf/dt)
```

其中 Δf 项反映温度状态，dΔf/dt 项反映温度梯度（热不均匀性），交叉项反映耦合效应。

---

## 7. PINN总体损失函数设计

### 7.1 完整损失函数

```
L_total = λ_d · L_data           (测量数据拟合)
        + λ_h · L_heat           (热传导方程残差)
        + λ_e · L_thermoelastic  (热弹性方程残差)
        + λ_s · L_sensor         (传感器物理模型残差)
        + λ_bc · L_boundary      (边界/初始条件)
        + λ_c · L_continuity     (层间连续性约束)
```

### 7.2 网络架构建议

```
输入: [t, T_env(t), dT_env/dt, 传感器内部温度读数]
  ↓
Branch 1 (温度场网络): → T̂(x,t)
Branch 2 (应力/变形网络): → σ̂(T), û(T)  
Branch 3 (漂移预测网络): → B̂(T, dT/dt), ŜF(T)
  ↓
输出: δω̂(t), δâ(t)  (预测的漂移误差)
```

### 7.3 可学习物理参数（逆问题）

以下参数可作为PINN的可学习变量（半监督逆问题）：

| 参数 | 物理含义 | 典型值 |
|------|---------|-------|
| k_E | 杨氏模量温度系数 | ~60 ppm/°C |
| α | 热膨胀系数 | ~2.6 ppm/°C |
| k | 有效热导率 | ~130 W/(m·K) |
| h_conv | 对流换热系数 | 取决于封装 |
| Δ(1/τ) | 阻尼各向异性 | 器件相关 |
| θ_τ | 主阻尼轴方位角 | 器件相关 |
| β₁, β₂ | 偏置温度系数 | 器件相关 |

---

## 8. 关键参考文献

### 热弹性物理模型
1. Hosseini-Pishrobat & Tatar, "Thermal stresses in multiring MEMS gyroscopes", J. Sound Vib., 2025
2. IEEE, "Modeling Temperature Effects in a MEMS Ring Gyroscope: Toward Physics-Aware Drift Compensation", IEEE J., 2025
3. Nature, "Thermoelastic damping in MEMS gyroscopes at high frequencies", Microsyst. Nanoeng., 2023
4. ScienceDirect, "An accurate thermoelastic model of HRG under varying temperatures", MSSP, 2022

### 传感器级解析模型
5. Zhou et al., "Analytical study and compensation for temperature drifts of MEMS accelerometer", Sens. Act. A, 2016
6. Zhou et al., "Analytical study of TCB and TCSF of MEMS accelerometers in closed-loop mode", Sens. Act. A, 2019
7. PMC, "Thermally induced packaging effects on frequency drift of MEMS resonant accelerometer", 2023

### 陀螺偏置物理模型
8. Prikhodko et al., "Compensation of drifts in high-Q MEMS gyroscopes using temperature self-sensing", Sens. Act. A, 2013
9. MDPI, "Consideration of Thermo-Vacuum Stability of a MEMS Gyroscope for Space Applications", Sensors, 2020
10. Zhuravlev, "Temperature drift of a hemispherical resonator gyro", Mech. Solids, 2018

### PINN + IMU
11. MDPI Sensors, "Vehicle State Estimation Combining PINN and UKF-M", 2023
12. arXiv, "TE-PINN: Quaternion-Based Orientation Estimation using Transformer-Enhanced PINN", 2024
13. arXiv, "MoRPI-PINN: Physics-Informed Framework for Mobile Robot Pure Inertial Navigation", 2025
14. APS, "PINNs for Orientation Estimation from IMU sensors", 2025
