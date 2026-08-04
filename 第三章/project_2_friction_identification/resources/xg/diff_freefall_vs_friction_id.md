# xg_freefall.xml vs xg_friction_id.xml 差异对比

## 总览

| | xg_freefall.xml | xg_friction_id.xml |
|---|---|---|
| 用途 | 自由落体验证（第二章） | 摩擦参数辨识（第三章） |
| 模型名 | `DOG_FREEFALL` | `XG_FRICTION_ID` |
| 核心思想 | 无地面、无摩擦，验证纯重力下角动量守恒 | 有地面、有接触，采集关节数据反向辨识摩擦力 |

---

## 差异明细

### 1. 接触开关

```xml
<!-- freefall -->
<flag contact="disable"/>

<!-- friction_id -->
<flag contact="enable"/>
```

freefall 完全关闭接触引擎；friction_id 需要地面接触来产生足端反力，关节才会有负载、才能表现出摩擦效应。

---

### 2. 默认几何体属性

```xml
<!-- freefall -->
<geom condim="3" contype="0" conaffinity="0"/>

<!-- friction_id -->
<geom condim="3" friction="0.8 0.005 0.0001" solref="0.035 1.1" solimp="0.9 0.92 0.015"/>
```

- freefall：所有 geom 默认不参与碰撞（contype/conaffinity = 0），整个模型就像在真空中
- friction_id：指定了**摩擦系数**和**接触求解器参数**，为足端与地面的碰撞做准备

---

### 3. 默认关节属性

```xml
<!-- freefall -->
<joint damping="0" armature="0.01"/>

<!-- friction_id -->
<joint damping="0" frictionloss="0" armature="0.01"/>
```

friction_id 多了一个 `frictionloss="0"` —— 库仑摩擦字段。初始值为 0，后续由 Python 脚本注入真值。freefall 不需要这个字段，因为它不辨识摩擦。

---

### 4. 地面

```xml
<!-- freefall -->
（无地面）

<!-- friction_id -->
<geom name="floor" type="plane" size="50 50 0.01"
      contype="1" conaffinity="1" friction="1.0 0.005 0.0001" condim="3"
      rgba="0.35 0.35 0.35 1"/>
```

freefall 没有地面 —— 机器狗在虚空中自由落体。friction_id 有地面 —— 机器狗站在上面做 chirp 激励和 trot 步态。

---

### 5. 光源

```xml
<!-- freefall -->
（无光源）

<!-- friction_id -->
<light name="spotlight" mode="fixed" directional="true"
       diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" pos="0 0 5" dir="0 0 -1"/>
```

friction_id 多了一个方向光，纯渲染用途，不影响物理。

---

### 6. 躯干 geom

```xml
<!-- freefall -->
<geom type="sphere" size="0.12" pos="0 0 0" rgba="0.75 0.75 0.75 0.5"/>

<!-- friction_id -->
<geom type="sphere" size="0.12" pos="0 0 0" contype="0" conaffinity="0" rgba="0.75 0.75 0.75 0.5"/>
```

friction_id 的躯干 geom 显式写了 `contype="0" conaffinity="0"`，因为它的 default 不再全局禁用碰撞，需要逐个排除不想参与碰撞的部件。

---

### 7. 连杆 geom（所有 HIP / KNEE 连杆几何体）

两个文件中所有非足端的 geom 视觉属性相同，但 friction_id 每个都显式加了：

```xml
contype="0" conaffinity="0"
```

原因同上 —— 只有足端需要碰地面，其余部位必须排除。

---

### 8. 足端 geom（最关键差异）

```xml
<!-- freefall -->
<geom type="sphere" size="0.02" rgba="1 0 0 0.5"/>      <!-- 无碰撞，半透明 -->

<!-- friction_id -->
<geom name="FR" type="sphere" size="0.02" contype="1" conaffinity="1" rgba="1 0 0 0.8"/>
```

| | freefall | friction_id |
|---|---|---|
| 名称 | 无 | `FR` `FL` `RR` `RL` |
| 碰撞 | 无（继承 default 的 0） | **有**（contype=1, conaffinity=1） |
| 颜色 | 半透明 (alpha=0.5) | 不透明 (alpha=0.8) |

---

### 9. 膝关节运动范围

```xml
<!-- freefall -->
<joint name="FAR_KNEE_JOINT" ... range="-2.723 0.602"/>

<!-- friction_id -->
<joint name="FAR_KNEE_JOINT" ... range="-2.723 -0.602"/>
```

| | 下限 | 上限 |
|---|---|---|
| freefall | -2.723 rad | **+0.602 rad** |
| friction_id | -2.723 rad | **-0.602 rad** |

freefall 的膝关节可以**过伸**到正角度（+0.602 rad，约 +34.5°），friction_id 限制在负角度（最大 -0.602 rad，约 -34.5°），更接近真实站立/步态运动范围。这个差异是故意的 —— freefall 允许更大范围来验证极端姿态，friction_id 限制在实际步态范围内保证数据质量。

---

## 不变的部分（两份文件完全相同）

- 身体拓扑结构（5 刚体 × 4 腿）
- 所有质量和惯性参数
- 所有关节位置和转轴
- 12 个执行器定义
- 仿真参数（步长 0.001s, Euler 积分器, Newton 求解器）
- home keyframe 预设姿态
