# Crazyflie 2.1 Digital Twin: Mathematical Modeling Equations

This document summarizes the equations used in the modeling of the Crazyflie 2.1 quadrotor digital twin simulation. It is formatted to provide a clear and concise representation of the underlying physics and dynamics for other mathematical or agentic LLMs.

## 1. 6-DOF Rigid Body Dynamics
The quadrotor state consists of a 13-dimensional state vector $\mathbf{x} = [\mathbf{p}, \mathbf{v}, \mathbf{q}, \boldsymbol{\omega}]^T$, representing position (NED frame), velocity (world frame), attitude (quaternion), and angular velocity (body frame). Let $m$ be the total mass and $\mathbf{I}$ the inertia tensor.

### Translational Dynamics
The translational acceleration in the world frame is determined by Newton's Second Law:
$$ \mathbf{\dot{v}} = \frac{1}{m} \left( \mathbf{R}(\mathbf{q}) \mathbf{F}_{\text{body}} \right) + \mathbf{g}_{\text{world}} $$
Where:
- $\mathbf{R}(\mathbf{q})$ is the rotation matrix derived from the quaternion $\mathbf{q}$.
- $\mathbf{F}_{\text{body}}$ is the total net non-gravitational force in the body frame.
- $\mathbf{g}_{\text{world}} = [0, 0, g]^T$ is the gravity vector in the NED world frame.

### Rotational Kinematics (Quaternion)
The quaternion derivative relates to the body angular velocity $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$:
$$ \mathbf{\dot{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix} $$
*(Using Hamilton product $\otimes$)*

### Rotational Dynamics (Euler's Equation)
The angular acceleration in the body frame is calculated as:
$$ \boldsymbol{\dot{\omega}} = \mathbf{I}^{-1} \big( \boldsymbol{\tau}_{\text{body}} - \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega}) \big) $$
Where $\boldsymbol{\tau}_{\text{body}}$ is the total net torque in the body frame.

## 2. Motor Dynamics
Brushed coreless DC motors are modelled with a first-order lag and empirical steady-state thrust curves for the input PWM signal $u \in [0, 1]$.

### Control Signal to Thrust
$$ F_{\text{target}} = \max(a \cdot u^2 + b \cdot u, 0) $$
Target angular velocity: $\omega_{\text{target}} = \sqrt{F_{\text{target}} / k_f}$

### First-Order Lag (Motor Angular Velocity)
Using a time constant $\tau_m$, the motor angular velocity $\omega_i$ undergoes exponential lag:
$$ \dot{\omega}_i = \frac{1}{\tau_m} (\omega_{\text{target}, i} - \omega_i) $$

### Resulting Force and Torque from Motor $i$
$$ \text{Thrust: } F_i = k_f \omega_i^2 $$
$$ \text{Reactive Torque: } M_i = -s_i k_m \omega_i^2 $$
where $s_i \in \{+1, -1\}$ handles CW / CCW rotor spin direction.

## 3. Motor Mixing and Body Wrench
The Crazyflie uses an X-configuration. With arm length $L$, the geometric displacement $d = \frac{L \sqrt{2}}{2}$ maps the thrusts ($F_i$) to body forces and torques:

$$ F_{z, \text{body}} = -\sum_{i=1}^4 F_i $$
$$ \tau_{x, \text{body}} = -d(F_1 + F_4 - F_2 - F_3) \quad \text{(Roll torque)} $$
$$ \tau_{y, \text{body}} = d(F_1 + F_2 - F_3 - F_4) \quad \text{(Pitch torque)} $$
$$ \tau_{z, \text{body}} = \sum_{i=1}^{4} M_i \quad \text{(Yaw torque via reactive differences)} $$
*(Note: Force and torque coefficient sums implicitly handle the geometric position of engines $M_1$ through $M_4$ in the body frame).*

## 4. Aerodynamics Model
Micro-UAV aerodynamic effects are modelled using predominantly linear regimes. Let $\mathbf{v}_{\text{body}} = \mathbf{R}^T \mathbf{v}_{\text{world}}$.

### Translational Drag
$$ \mathbf{F}_{\text{drag}} = - \begin{bmatrix} k_{xy} & 0 & 0 \\ 0 & k_{xy} & 0 \\ 0 & 0 & k_z \end{bmatrix} \mathbf{v}_{\text{body}} $$

### Rotational Drag
$$ \boldsymbol{\tau}_{\text{drag}} = -k_{\text{rot}} \boldsymbol{\omega} $$

### Ground Effect (Thrust Amplification)
When operating near the ground ($h = -z < h_{\text{ceiling}}$), the thrust is dynamically amplified:
$$ F_{\text{boosted}} = F_{\text{base}} \cdot \left[1 + (m_{\max} - 1) \left(1 - \frac{h}{h_{\text{ceiling}}}\right)\right] $$

## 5. Sensor Models

### IMU Model (Accelerometer and Gyroscope)
Sensors integrate white noise ($N \sim \mathcal{N}(0, \sigma^2)$) and a random walk bias process ($b_k = b_{k-1} + \mathcal{N}(0, \sigma_{walk}^2)$).

**Accelerometer (Measures Specific Force):**
$$ \mathbf{a}_{\text{meas}} = \left(\frac{\mathbf{F}_{\text{body}}}{m}\right) - \mathbf{R}^T \mathbf{g}_{\text{world}} + \mathbf{b}_{\text{accel}} + \mathbf{n}_{\text{accel}} $$
*(During hover at pitch/roll = 0, the accelerometer measures $-\mathbf{g}$ locally).*

**Gyroscope:**
$$ \boldsymbol{\omega}_{\text{meas}} = \boldsymbol{\omega} + \mathbf{b}_{\text{gyro}} + \mathbf{n}_{\text{gyro}} $$

### Barometer Model
Models geometric altitude $h$ directly mapping to pressure $P$ roughly using the International Standard Atmosphere (ISA) formula, adding sensor bias and static noise:
$$ P_{\text{true}}(h) = P_0 \left( 1 - \frac{L \cdot h}{T_0} \right)^{\frac{g}{R_{gas} L}} $$
$$ P_{\text{meas}} = P_{\text{true}}(h) + b_{\text{baro}} + \mathcal{N}(0, \sigma_P^2) $$
The inverse formula provides the measured altitude estimate.
