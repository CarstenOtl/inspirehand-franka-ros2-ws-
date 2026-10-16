# **An Overview of Null Space Projections for Redundant, Torque Controlled Robots**

Alexander Dietrich<sup>1</sup> , Christian Ott<sup>1</sup> , and Alin Albu-Sch¨affer<sup>1</sup><sup>_,_2</sup>

> 1Institute of Robotics and Mechatronics, German Aerospace Center (DLR)

> 2Technische Universit¨at M¨unchen (TUM), Germany

Contact: `Alexander.Dietrich@dlr.de`

January 8, 2015

## Abstract
_One step on the way to approach human performance in robotics is to provide joint torque sensing and control for better interaction capabilities with the environment, and a large number of actuated degrees of freedom (DOF) for improved versatility. However, the increasing complexity also raises the question of how to resolve the kinematic redundancy which is a direct consequence of the large number of DOF. Here we give an overview of the most practical and frequently used torque control solutions based on null space projections. Two fundamental structures of task hierarchies are reviewed and compared, namely the successive and the augmented method. Then the projector itself is investigated in terms of its consistency. We analyze static, dynamic, and the new concept of stiffness consistency. In the latter case, stiffness information is used in the pseudoinversion instead of the inertia matrix. In terms of dynamic consistency, we generalize the weighting matrix from the classical operational space approach and show that an infinite number of weighting matrices exist to obtain dynamic consistency. In this context we also analyze another dynamically consistent null space projector with slightly different structure and properties. The redundancy resolutions are finally compared in several simulations and experiments. A thorough discussion of the theoretical and empirical results completes this survey._

## 1 Introduction
The impressive performance of a human being is substantially due to its versatility. The large number of degrees of freedom (DOF) allows to adapt to a variety of environments and several simultaneous objectives. Consider a service task like setting a table, for example. Beside the main pick-and-place task a large number of objectives have to be accounted for additionally: Collisions have to be avoided,

the balance has to be held, the environment has to be observed permanently, and unexpected disturbances have to be compensated for. All these subtasks have to be fulfilled in some form or another, but there are usually more important and less important aspects such that a hierarchy among the tasks can be established.

In robotics, the most frequently applied method to resolve such a kinematic redundancy is doubtless the null space projection technique developed in the 1980s [Khatib, 1987; Nakamura et al., 1987; Siciliano and Slotine, 1991]. The concept is based on a hierarchical arrangement of the involved tasks and can be interpreted as an instantaneous, local optimization. The top priority task is executed employing all capabilities of the robotic system. The second priority task is then applied to the null space of the top priority task. In other words, the task on the second level is executed as good as possible without disturbing or interfering with the first level. The task on level three is then executed without disturbing the two higher priority tasks, and so forth. Today these techniques are standard tools in kinematic control [Baerlocher and Boulic, 2004; Nakanishi et al., 2008; Antonelli et al., 2009; Decr´e et al., 2009; Sugiura et al., 2010; Kanoun et al., 2011; Lee et al., 2011] and dynamic control [Albu-Sch¨affer et al., 2003; Khatib et al., 2004; Sentis and Khatib, 2005; Nakanishi et al., 2008; Mansard et al., 2009; Dietrich et al., 2012b; Sadeghian et al., 2013].

A clear overview and comparison of practical null space projections for torque control has not been given so far. We provide such a survey in a unified framework and integrate seminal results from the robotics community, while extending the knowledge base at several places with new insights. The motivation for this paper was that existing works on null space based redundancy resolutions only cover parts for a complete survey such that their results have to be combined and condensed for an elaborate overview: Antonelli [Antonelli, 2009] compares two different kinds of strictnesses in the hierarchy for kinematic control, namely

the _successive_ [Dietrich et al., 2012b] and the _augmented_ [Siciliano and Slotine, 1991; Sentis and Khatib, 2005] null space projections. In our survey, these two basic domains are analyzed concisely for torque controlled robots. Apart from this overall structure of the task hierarchy, the null space projector itself has essential inherent properties in terms of its consistency. Based on the weighting matrix in the pseudoinversion of the Jacobian matrix [Doty et al., 1993], we compare _static_ , _dynamic_ , and the novel idea of _stiffness consistency_ of the projections. The type of consistency is closely related to the question of using the inertia matrix [Khatib, 1987; Sentis and Khatib, 2005; Featherstone, 2010; Sadeghian et al., 2013] or other, possibly constant weighting matrices in the pseudoinversion [Baillieul et al., 1984; Hollerbach and Suh, 1987; Albu-Sch¨affer et al., 2003; Dietrich et al., 2012a]. Dynamically consistent projectors are investigated in particular since they are probably the most common choices in torque control. Elaborate comparisons among the subclass of inertia-based null space projectors have been performed in the literature [Nakanishi et al., 2008; Peters et al., 2008; Hollerbach and Suh, 1987], but most authors conclude that for high performance, an accurate model of the inertia matrix is necessary which is both difficult to obtain and computationally very expensive. For this reason the experimental comparison between dynamically consistent approaches utilizing the inertia matrix [Khatib, 1987] and statically consistent techniques _without_ explicit use of the inertia matrix [Albu-Sch¨affer et al., 2003] are of high relevance in robotics. Especially when considering the implementation on real hardware, we show that theoretically superior techniques actually lose most of their benefits.

The main contribution of this paper is the comprehensive overview and discussion of different null space projection techniques for the particular case of torque control. Furthermore we interprete the weighting matrix in the popular dynamically consistent null space projector by Khatib [Khatib, 1987, 1995] as a special case of an infinite number of dynamically consistent weighting matrices. This analysis contributes to a better understanding of dynamic consistency in general. We analyse a further kind of dynamically consistent null space projectors which do not derive from the standard procedure for torque control but from acceleration-based robot control, yet they share most of the properties with the classical solution. Moreover we introduce the new idea of stiffness consistency. Instead of employing knowledge about the inertia distribution in the null space projector computation, stiffness information is utilized to obtain useful new features in the redundancy resolution. The comparison of the null space projectors is supported by extensive simulations and experiments on a real torque controlled robot. As a result of this work, an expedient overview of torque control null space projectors is

provided with which the operator of the robot can make his choice depending on the application case and the resources.

## 2 Strictness of the Hierarchy
Consider a manipulator with _n_ DOF and _r_ task coordinates which are defined by

```
xi = f i(q) ∈Rmi    (1)
```

for 1 _≤ i ≤ r_ . The dimension of task _i_ is _mi ≤ n_ . The differential mappings from joint velocities to task velocities are given by the Jacobian matrices **_J_** _i_ ( **_q_** ) _∈_ R<sup>_mi×n_</sup> with

```
˙xi = Ji(q) ˙q ,
Ji(q) = ∂f i(q)
∂q
.    (2)
```

In the following, **_J_** _i_ ( **_q_** ) is assumed be non-singular, hence of full row-rank. Dealing with singular matrices or changing rank requires additional treatment [Deo and Walker, 1995], both in kinematic control [Chiaverini, 1997] and torque control [Dietrich et al., 2012a,b]. Since the primary task ( _i_ = 1) has dimension _m_ 1 _< n_ , a kinematic redundancy of _n − m_ 1 DOF remains to accomplish subtasks in its null space. The hierarchy is defined such that _i_ = 1 is top priority and _ia < ib_ implies that _ia_ is located higher in the priority order than _ib_ .

### 2.1 Successive Projections
In the successive null space projection [Antonelli, 2009; Dietrich et al., 2012b] a task torque **_τ_** 2 _∈_ R<sup>_n_</sup> on the second priority level is projected into the null space of the main task ( _i_ = 1) by applying

```
τ p
2 = N suc
2 (q)τ 2 ,    (3)
```

where **_τ_**<sup>p</sup> 2<sup>_∈_R</sup><sup>_n_</sup> is the projected torque that does not inter- fere with the main task. The successive null space projector **_N_**<sup>suc</sup> 2<sup>(</sup><sup>**_q_**</sup>) is obtained by evaluating

```
N suc
2 (q) = I −J1(q)T (J1(q)#)T ,    (4)
```

wherein _{}_<sup>#</sup> represents the generalized inverse and **_I_** is the identity matrix. Analogous to (3), the remaining tasks in the hierarchy (2 _< i ≤ r_ ) can be implemented by

```
τ p
i = N suc
i
(q)τ i    (5)
```

with the null space projectors obtained via the recursive, _successive_ algorithm

```
N suc
i
(q) = N suc
i−1(q)
I −Ji−1(q)T (Ji−1(q)#)T
.    (6)
```

One receives the final control torque by adding up the main task torque and all projected torques to

```
τ = τ 1 +
r
X
i=2
τ p
i .    (7)
```

### 2.2 Augmented Projections
The augmented approach [Siciliano and Slotine, 1991] is identical to the successive projection on the first null space level (3)–(4). From the third level on, the projected torque is given by

```
τ p
i = N aug
i
(q)τ i ,    (8)
```

```
N aug
i
(q) = I −Jaug
i−1(q)T (Jaug
i−1(q)#)T .    (9)
```

The _augmented_ Jacobian matrix **_J_**<sup>aug</sup> _i−_ 1<sup>(</sup><sup>**_q_**</sup>) takes all higher priority Jacobian matrices into account:

```
Jaug
i−1(q) =
J1(q)
J2(q)
...
Ji−1(q)
 .    (10)
```

The final control torque is obtained via (7) again by using (8) instead of (5) now. The direct implementation of (9) is computationally expensive due to the large number of rows in **_J_**<sup>aug</sup> _i−_ 1<sup>(</sup><sup>**_q_**).</sup> Usually recursive algorithms [Siciliano and Slotine, 1991; Baerlocher and Boulic, 1998; Sentis and Khatib, 2005] are applied to reduce the numerical effort:

```
N aug
1
= I ,    (11)
ˆ
Ji(q) = Ji(q)N aug
i
(q)T ,    (12)
N aug
i
(q) = N aug
i−1(q)
I −ˆ
Ji−1(q)T ( ˆ
Ji−1(q)#)T    (13)
```

Herein **_J_**<sup>**ˆ**</sup> _i_ ( **_q_** ) _∈_ R<sup>_mi×n_</sup> describes the Jacobian matrix of level _i_ projected into the null space of all higher priority tasks.

In fact, this additional recursive step (12) is the only difference between the successive and the augmented approach. A direct comparison between the two types of hierarchy strictnesses will be given in Section 5.1.

## 3 Consistency of the Projections
While the preceding section investigated the overall structure of the hierarchy, the _consistency_ determines how the null space itself is defined in terms of properties and shape. Prior to that analysis, the dynamic equations of the robot and the pseudoinverse of a matrix are briefly reviewed in Section 3.1.

### 3.1 Dynamic Equations and Pseudoinverse
The dynamic equations of a robot with _n_ DOF can be written as

```
M(q)¨q + C(q, ˙q) ˙q + g(q) = τ + τ ext .    (14)
```

The inertia matrix **_M_** ( **_q_** ) _∈_ R<sup>_n×n_</sup> is symmetric, positive definite, and it depends on the joint configuration **_q_** _∈_ R<sup>_n_</sup> . Gravity torques are taken into account by **_g_** ( **_q_** ) _∈_ R<sup>_n_</sup> , and Coriolis/centrifugal effects are represented by **_C_** ( **_q_** _,_ **_q_** ˙ ) ˙ **_q_** _∈_ R<sup>_n_</sup> . The joint torques are described by **_τ_** _∈_ R<sup>_n_</sup> , and external forces are denoted by **_τ_**<sup>ext</sup> _∈_ R<sup>_n_</sup> . In these notations we use the terms _joint torques_ and _external forces_ since robots are rather equipped with revolute joints than prismatic joints, and external loads are usually applied in terms of forces instead of torques. However, the extension to generalized joint forces (including forces and torques) as well as generalized external forces (including forces and torques) can be made without loss of generality. In the following analysis, we will set the control input to

```
τ = τ ′ + C(q, ˙q) ˙q + g(q)    (15)
```

to compensate for the Coriolis/centrifugal terms and gravitational effects such that

```
M(q)¨q = τ ′ + τ ext .    (16)
```

As of now the “new” control input **_τ_**<sup>_′_</sup> will be used. Notice that all conclusions in the subsequent sections are also valid without the compensation (15). Nevertheless, neglecting **_C_** ( **_q_** _,_ **_q_** ˙ ) ˙ **_q_** and **_g_** ( **_q_** ) by means of (16) improves the readability and highlights the relevant aspects better.

In Section 2 the generalized inverse _{}_<sup>#</sup> was used but it was not specified. A generalized inverse **_A_**<sup>#</sup> of a full row rank matrix **_A_** _∈_ R<sup>_m×n_</sup> with _m < n_ has to satisfy the criterion

```
AA# = I    (17)
```

for _right inverses_ . One can find an infinite number of generalized inverses that meet (17). As of now, the notation _{}_<sup>**_W_**+</sup> is used instead of _{}_<sup>#</sup> to disambiguate the inverse by the weighting matrix **_W_** _∈_ R<sup>_n×n_</sup> . Hence one can formulate

```
AW + = W −1AT (AW −1AT )−1 ,    (18)
```

which fulfills (17) as long as the inversion on the right is feasible. The term **_AW_**<sup>_−_1</sup> **_A_**<sup>_T_</sup> has to be of rank _m_ , and **_W_** must be invertible. The use of such generalized inverses is very common in robotics [Doty et al., 1993], for example in inverse kinematics. In the following, the effects of the weighting matrix are clarified and classified into three different types of torque control projection consistencies. That analysis is performed on a two-level system for the sake of simplicity, yet all statements can be transferred to more complex hierarchies without loss of generality. A distinction between successive and augmented projection does not have to be made here since **_N_** 2( **_q_** ) = **_N_**<sup>suc</sup> 2<sup>(</sup><sup>**_q_**) =</sup><sup>**_N_**aug</sup> 2 ( **_q_** ).

```
2 (q) = N aug
2
(q).
3.2
```

### 3.2 Static Consistency
**Definition 1.** _A null space projector_ **_N_** _j_ ( **_q_** ) _∈_ R<sup>_n×n_</sup> _is said to be “statically consistent” if a subtask does not generate_

_interfering forces in the operational spaces of all higher priority tasks in any static equilibrium. The condition_

```
(Ji(q)W +)T N j(q) = 0    (19)
```

_for i < j must hold in any steady state with_ **_q_** ˙ = **_q_** ¨ = **0** _._

In order to show that we set

```
τ ′ = N 2(q)τ 2    (20)
```

and consider a static scenario where the external forces are only given by the reaction forces **_F_**<sup>react</sup> 1 which are exerted on the robot by clamping the robot in the main task space, e. g. by clamping the end-effector in case of a Cartesian main task of the end-effector. Then the external force is given by

```
τ ext = J1(q)T F react
1
.    (21)
```

Inserting (20) and (21) into the quasi-static version of (16), and reorganizing the terms, yields

```
−J1(q)T F react
1
= N 2(q)τ 2 .    (22)
```

From (17) we can conclude that the multiplication by ( **_J_** 1( **_q_** )<sup>**_W_**+</sup> )<sup>_T_</sup> from the left will lead to

```
F react
1
= −(J1(q)W +)T N 2(q)τ 2    (23)
= −(J1(q)W +)T (I −J1(q)T (J1(q)W +)T )
|
{z
}
= 0
τ 2    (24)
```

for any admissible weighting matrix **_W_** . Thus the conditions of _static consistency_ are met according to Definition 1. The effect of the null space task **_τ_** 2 onto the main task force **_F_**<sup>react</sup> 1 is zero [Albu-Sch¨affer et al., 2003], hence no main task acceleration will be generated in this equilibrium such that **_x_** ¨ 1 = **0** holds. This result can also be interpreted as the confirmation of the equilibrium, in which no torque from the null space disturbs the main task anymore.

The simplest weighting matrix is

```
W = I ,    (25)
```

so that for **_A_** = **_J_** 1( **_q_** ) one can write (18) as

```
J1(q)I+ = J1(q)+ = J1(q)T (J1(q)J1(q)T )−1 .    (26)
```

In the notation of this so-called _Moore-Penrose pseudoinverse_ , the identity in the superscript is often omitted. Compared to other weighting matrices, this choice is computationally cheap and also has further advantages due to its reduced complexity. This null space projector can be interpreted from a geometric point of view [Dietrich et al., 2012c], for example, and damped least-squares techniques can be applied easily [Deo and Walker, 1995].

### 3.3 Dynamic Consistency
The property of static consistency is shared by all null space projectors as described in the previous section, independent of the weighting matrix. But apart from static consistency, specific weighting matrices offer additional beneficial properties such as the so-called dynamic consistency treated in this section. The main difference is that static consistency only guarantees that the hierarchy levels do not interfere in a steady state, while dynamic consistency guarantees _additionally_ that they also do not interfere during the transient into this steady state.

**Definition 2.** _A null space projector_ **_N_** _j_ ( **_q_** ) _∈_ R<sup>_n×n_</sup> _is said to be “dynamically consistent” [Khatib, 1995] if it is “statically consistent” and if a subtask never generates accelerations in the operational spaces of all higher priority tasks. The condition_

```
Ji(q)M(q)−1N j(q) = 0    (27)
```

_for i < j must be fulfilled at any time._

The dynamics (16) can be rewritten as

```
¨x1 = J1(q)M(q)−1τ ′ + J1(q)M(q)−1τ ext + ˙J1(q) ˙q
|
{z
}
p1(q, ˙q, τ ext)    (28)
```

after projection into the main task directions defined by **_x_** 1. The term **_p_** 1( **_q_** _,_ **_q_** ˙ _,_ **_τ_**<sup>ext</sup> ) is introduced for the sake of simplicity. Applying the control input

```
τ ′ = J1(q)T F 1 + N 2(q)τ 2    (29)
```

with the main task force **_F_** 1 _∈_ R<sup>_m_1</sup> modifies (28) to

```
¨x1 = p1(q, ˙q, τ ext)+Λ1(q)−1F 1+J1(q)M(q)−1N 2(q)τ 2 ,    (30)
```

**_x_** ¨ 1 = **_p_** 1( **_q_** _,_ **_q_** ˙ _,_ **_τ_**<sup>ext</sup> )+ **Λ** 1( **_q_** )<sup>_−_1</sup> **_F_** 1 + **_J_** 1( **_q_** ) **_M_** ( **_q_** )<sup>_−_1</sup> **_N_** 2( **_q_** ) **_τ_** 2 _,_ (30) where the main task inertia is defined as

```
Λ1(q) = (J1(q)M(q)−1J1(q)T )−1 .    (31)
```

The direct effect of the second level torque **_τ_** 2 _∈_ R<sup>_n_</sup> on the main task acceleration **_x_** ¨ 1 is determined by the coefficient of **_τ_** 2, i. e. (27) must be fulfilled for _i_ = 1 and _j_ = 2 to eliminate any effects of the lower priority task on the main task acceleration. Then the conditions of _dynamic consistency_ are met according to Definition 2. An intuitive interpretation of (27) is that the projector decouples the inertias on all priority levels.

#### 3.3.1 Configuration Dependent Weighting Matrix _W_ ( _q_ ) that uses the Inertia Matrix
Khatib [Khatib, 1987] has shown that the weighting matrix

```
W (q) = M(q)    (32)
```

fulfills (27) and the corresponding generalized inverse minimizes the instantaneous kinetic energy of the manipulator. Another choice has been proposed by Park [Park, 1999], which has the form

```
W (q) = J1(q)T J1(q) + M(q)Y 1(q)T Y 1(q)M(q) , (33)
```

where **_Y_** 1( **_q_** ) _∈_ R<sup>(</sup><sup>_n−m_1)</sup><sup>_×n_</sup> is a matrix that spans the null space of **_J_** 1( **_q_** ). In fact, an infinite number of configuration dependent weighting matrices **_W_** ( **_q_** ) exist that feature dynamic consistency. For a general formulation, the Jacobian matrix **_J_** 1( **_q_** ) is decomposed via singular value decomposition [Maciejewski and Klein, 1989] such that

```
J1(q) = U 1(q)S1(q)V 1(q)T ,    (34)
```

where **_U_** 1( **_q_** ) _∈_ R<sup>_m_1</sup><sup>_×m_1</sup> and **_V_** 1( **_q_** ) _∈_ R<sup>_n×n_</sup> are orthogonal matrices, and **_S_** 1( **_q_** ) _∈_ R<sup>_m_1</sup><sup>_×n_</sup> is a rectangular diagonal matrix containing the singular values **_σ_** 1 to **_σ_** _m_ 1. The null space can be geometrically interpreted easily when considering

```
V 1(q) =
X1(q)T , Y 1(q)T
.    (35)
```

Here, the _m_ 1 rows in **_X_** 1( **_q_** ) _∈_ R<sup>_m_1</sup><sup>_×n_</sup> span the range space of **_J_** 1( **_q_** ), while the _n−m_ 1 rows in **_Y_** 1( **_q_** ) span its null space. The orthogonality **_X_** 1( **_q_** ) **_Y_** 1( **_q_** )<sup>_T_</sup> = **0** holds. Inspired by (33), one can formulate a general rule for the weighting matrix **_W_** ( **_q_** ) that always fulfills the requirements of dynamic consistency:

```
W (q) = X1(q)T X1(q)BX + BY Y 1(q)T Y 1(q)M(q) .    (36)
```

The proof is provided in the Appendix. Note that **_W_** ( **_q_** ) has to be nonsingular to apply the standard algorithm (18) where **_W_** ( **_q_** )<sup>_−_1</sup> is used. Hence

```
(37)
```

must hold. Note that (37) is a necessary but not sufficient condition for the existence of **_W_** ( **_q_** )<sup>_−_1</sup> . However, in the Appendix it is also shown that the condition on the rank of **_BX_** can even be dropped when using another formulation than the one based on the pseudoinversion (18). With the knowledge of the general formulation, the weighting matrices of Khatib (32) and Park (33) can be regarded as special cases of (36) in fact:

```
BY = M(q)
```

Khatib [Khatib, 1987] found out that only one pseudoinverse satisfies (27). From that and the proof in the Appendix we can conclude that any weighting matrix (36) leads to the identical pseudoinverse which minimizes the instantaneous kinetic energy of the manipulator. With this

general formulation of **_W_** ( **_q_** ) the same null space projector results, that dynamically decouples the priority levels by block-diagonalizing the inertia matrix. This decoupling of the level-related inertias is an intuitive interpretation of Definition 2 as demonstrated in [Dietrich et al., 2013], for example. The formulation (36) contributes to a better understanding of dynamic consistency.

This null space projector has been shown to be load independent [Featherstone and Khatib, 1997]. Changing the load inertia or projected/reflected inertia on the higher priority levels does not result in a different null space projector. Let us consider such an additional load or modified reflected inertia **_L_** 1 _∈_ R<sup>_m_1</sup><sup>_×m_1</sup> and the respective, altered joint inertia matrix

```
M ⊕(q) = M(q) + J1(q)T L1J1(q) .    (38)
```

Then the equality

```
N 2(q) = I −J1(q)T (J1(q)M(q)+)T    (39)
= I −J1(q)T (J1(q)M ⊕(q)+)T    (40)
```

holds. Load independence allows to ignore loads in the control law. Their estimation or measurement can be avoided and using such a null space projector decouples internal motions from load-dependent influences [Featherstone and Khatib, 1997]. The invariance of the load can also be seen in the fact that (36) only requires knowledge of the inertia matrix applied to the null space **_Y_** 1( **_q_** ) and not necessarily to the range space **_X_** 1( **_q_** ), see also the Appendix.

#### 3.3.2 Arbitrary Weighting Matrix _W_
Indeed, another interesting type of dynamically consistent torque control null space projectors can be formulated, which originates from an acceleration-based approach:

```
N 2(q) = M(q)
I −J1(q)W +J1(q)
M(q)−1 .    (41)
```

The proof for dynamic consistency of (41) is provided in the Appendix. The premultiplication of **_M_** ( **_q_** ) ensures compliance with (27) and the multiplication by **_M_** ( **_q_** )<sup>_−_1</sup> from the right meets the idempotence requirement **_N_** 2( **_q_** ) = **_N_** 2( **_q_** ) **_N_** 2( **_q_** ). The major difference of (41) is that the null space projection is performed on acceleration level as it can be clearly seen in (41) in the middle term. If one considers **_τ_**<sup>p</sup> 2<sup>=</sup><sup>**_N_**2(</sup><sup>**_q_**)</sup><sup>**_τ_**2</sup> in combination with (41), the sec- ondary task torque is initially transformed into a joint acceleration through the multiplication by **_M_** ( **_q_** )<sup>_−_1</sup> . Then a (static) projection into the null space of the main task Jacobian matrix is performed as in standard kinematic robot control. Afterwards, this solution on acceleration level is transformed back into joint torques via **_M_** ( **_q_** ). The general idea of this procedure _torque → acceleration → null space_

```
BY = I
```

_projection → torque_ is intuitive and has been frequently implemented and analyzed before [Hollerbach and Suh, 1987; Peters et al., 2008].

The simplest choice for the weighting matrix is again **_W_** = **_I_** from Section 3.2. Moreover, due to the standard Moore-Penrose pseudoinversion in the middle, singularityrobust techniques such as [Dietrich et al., 2012a] can be applied easier to preserve continuity of the control law. This projector can also be computed in a recursive way to reduce the numerical effort. The adaptation of (11)–(13) to this case has the form

```
N aug,s
1
= I ,    (42)
ˆ
Ji(q) = Ji(q)N aug,s
i
(q)T ,    (43)
N aug,s
i
(q) = N aug,s
i−1 (q)
I −ˆ
Ji−1(q)+ ˆ
Ji−1(q)    (44)
N aug
i
(q) = M(q)N aug,s
i
(q)M(q)−1 .    (45)
```

The matrices **_N_**<sup>aug,s</sup> _i_ ( **_q_** ) _∈_ R<sup>_n×n_</sup> are auxiliary, statically consistent null space projectors on acceleration level, which are upgraded to dynamic consistency in (45). The implementation of (42)–(45) has basically similar properties as the solutions in Section 3.3.1: Dynamic consistency and the idempotence criterion **_N_**<sup>aug</sup> _i_ ( **_q_** ) = **_N_**<sup>aug</sup> _i_ ( **_q_** ) **_N_**<sup>aug</sup> _i_ ( **_q_** ) are fulfilled. However, load independency [Featherstone and Khatib, 1997] is not provided by this projector in general. Also note that for **_W_** = **_M_** ( **_q_** ), (41) yields the projector from Section 3.3.1.

```
i
(q) = N aug
i
(q)N aug
i
```

task by minimizing active regulation of the main task by exploiting the springs.

The dynamics (14) for constant external forces are extended by an additional joint spring **_k_** ( **_q_** _,_ **_q_** 0) _∈_ R<sup>_n_</sup> such that

```
M(q)¨q + C(q, ˙q) ˙q + g(q) + k(q, q0) = τ + τ ext ,    (46)
```

and **_q_** 0 _∈_ R<sup>_n_</sup> is the equilibrium configuration where the spring counterbalances the graviational load and the external forces.

**Definition 3.** _A null space projector_ **_N_** _j_ ( **_q_** 0) _∈_ R<sup>_n×n_</sup> _is said to be “stiffness consistent” if it is “statically consistent” and if a subtask does not cause static deviations in the operational spaces of all higher priority tasks. These higher prioritized tasks are executed by springs with positive definite stiffness matrix_

```
K(q0) = ∂k(q, q0)
∂q
q=q0
,    (47)
```

_where_ **_k_** ( **_q_** _,_ **_q_** 0) _∈_ R<sup>_n_</sup> _is a joint spring with equilibrium configuration_ **_q_** = **_q_** 0 _. The condition_

```
Ji(q0)K(q0)−1N j(q0) = 0    (48)
```

_for i < j must hold locally around the steady state_ **_q_** = **_q_** 0 _with_ **_q_** ¨ = **_q_** ˙ = **0** _._

In this equilibrium **_q_** 0 the linearizations

```
klin(q, q0) = k(q0) + ∂k(q, q0)
∂q
q=q0
(q −q0)    (49)
= k(q0) + K(q0)∆q ,    (50)
glin(q, q0) = g(q0) + ∂g(q)
∂q
q=q0
(q −q0) ,    (51)
= g(q0) + G(q0)∆q    (52)
```

### 3.4 Stiffness Consistency
An increasing number of parallel elastic actuators (PEAs) is encountered in the fields of prostheses, exoskeletons and rehabilitation [Dollar and Herr, 2008; Winfree et al., 2011; Haeufle et al., 2012; Grimmer et al., 2012], among others. Mounting mechanical springs in parallel to the motors allows to downsize the actuators because gravitational loads can be counterbalanced by the passive elements. Energy efficiency can be drastically improved that way, both from a static point of view (gravity compensation) and from a dynamic perspective (energy-efficient cyclic motions). The research group of Herr has recently achieved impressive results in the field of active prostheses with additional passive elements where the principles of biomechanics and neural control are combined to design new devices [Au and Herr, 2009].

Consider a scenario where a main task is statically accomplished by such a set of parallel mechanical springs, e. g. to keep the end-effector at a location by pre-adjusting the joints and (possibly variable) springs such that no motor power is required to maintain the main task configuration. The so-called _stiffness consistent_ null space projector can then be used to simultaneously accomplish a secondary

can be evaluated where **_K_** ( **_q_** 0) _∈_ R<sup>_n×n_</sup> is the positive definite stiffness matrix in the equilibrium, **_G_** ( **_q_** 0) _∈_ R<sup>_n×n_</sup> describes the local, linear gravity behavior, and ∆ **_q_** = **_q_** _−_ **_q_** 0. At **_q_** = **_q_** 0, the counterbalance **_k_** ( **_q_** 0) = _−_ **_g_** ( **_q_** 0)+ **_τ_**<sup>ext</sup> holds for constant external forces. Then the quasi-static version of the dynamics (46) with

```
τ = N 2(q0)τ 2    (53)
```

yields

```
K(q0)∆q = −G(q0)∆q + N 2(q0)τ 2 .    (54)
```

Locally around the equilibrium the differential mapping (2) can be used to obtain

```
∆x1 = J1(q0)K(q0)−1 (−G(q0)∆q + N 2(q0)τ 2)    (55)
```

<!-- Start of picture text -->
Level 2<br>Level 4<br>Level 1<br>Level 3<br>y<br>g<br>x<br>z<br><!-- End of picture text -->

Figure 1: Simulation model of a planar, four DOF system. The links are connected via revolute joints. Each link is modeled by a point mass of 1 kg that is placed in the middle of a bar with length 0.5 m. The dynamics are simulated using _g_ = 9 _._ 81 m _/_ s<sup>2</sup> .

which has clear similarities to (30). If the weighting matrix

```
W = K(q0)    (56)
```

is chosen, then the main task does not experience any disturbance by the lower priority task **_τ_** 2, hence ∆ **_x_** 1 = **0** . In other words, the contribution of the springs on the main task can be preserved by this choice for the null space projector and Definition 3 is met.

Any spring can be used for stiffness consistent null space projections, for example one with nonlinear spring torque of the form **_k_** ( **_q_** _,_ **_q_** 0 _,_ **_σ_** ) _∈_ R<sup>_n_</sup> , where **_σ_** _∈_ R<sup>_n_</sup> is the stiffness adjuster of a variable stiffness mechanism.

## 4 Simulations and Experiments
The section will provide simulations and experiments to demonstrate the properties of the null space projectors. In the first simulation of Section 4.1, an extensive comparison between successive and augmented null space projections as well as statically consistent and dynamically consistent redundancy resolutions is made. The second simulation shows the properties of the novel stiffness consistent null space projector in comparison to common statically consistent and dynamically consistent redundancy resolutions. In Section 4.2 the null space projectors are applied to a real torque controlled 7 DOF manipulator.

### 4.1 Simulations
The first simulation shows the theoretical properties of the presented null space projections on a planar _n_ = 4 DOF manipulator, see Fig. 1 for the simulated model. The task hierarchy is designed with the following levels:

1. Level ( _m_ 1 = 1): translational Cartesian impedance at the TCP (tool center point) in _x_ -direction,

Table 1: Controller gains for the simulations and experiments; (* additional integrator for zero steady-state error)

|**Gain**|**Sim. 1**|**Sim. 2**|**Experiment**|
|---|---|---|---|
|**_K_**1|800 <sup>N</sup><br>m<br>|**0**|diag(1200_,_1200_,_1200) <sup>N</sup><br>m|
|**_D_**1|60 <sup>Ns</sup><br>m|**0**|damping ratios set to 0.9|
|**_K_**2|800 <sup>N</sup><br>m<br>|200 <sup>Nm</sup><br>rad <sup>,*</sup><br>|diag(60_,_60_,_60) <sup>Nm</sup><br>rad|
|**_D_**2|60 <sup>Ns</sup><br>m|10 <sup>Nms</sup><br>rad|damping ratios set to 0.9|
|**_K_**3|150 <sup>Nm</sup><br>rad<br>|-|diag(20_, . . .,_20) <sup>Nm</sup><br>rad<br>|
|**_D_**3|4 <sup>Nms</sup><br>rad<br>|-|diag(3_, . . .,_3) <sup>Nms</sup><br>rad|
|**_K_**4|100 <sup>Nm</sup><br>rad<br>|-|-|
|**_D_**4|4 <sup>Nms</sup><br>rad|-|-|

```
rad
K4
100 Nm
rad
-
-
D4
4 Nms
rad
-
-
```

```
rad
D3
4 Nms
rad
-
```

```
K3
150 Nm
rad
-
```

```
rad
D2
60 Ns
m
10 Nms
rad
```

```
K2
800 N
m
200 Nm
rad ,*
```

2. Level ( _m_ 2 = 1): translational Cartesian impedance at the TCP in _y_ -direction,

3. Level ( _m_ 3 = 1): rotational Cartesian impedance at the TCP about the _z_ -axis,

4. Level ( _m_ 4 = 4): complete joint impedance.

Since<sup>�4</sup> _i_ =1<sup>_mi_=7</sup><sup>_>n_andthetaskspartiallyconflict</sup> with each other, not all of them can be accomplished to full extent. The controller gains are specified in Table 1. The regulation case and its transient responses are considered in the following. Fig. 2 depicts the step responses for five different implementations. Additionally, the solution without any null space projection is plotted as well. That means that the control torques from the individual priority levels are directly applied without being processed by any null space projectors at all, i. e. they are simply added. Thus, all tasks compete with each other without a proper hierarchy.

All augmented methods reach zero steady-state errors on the first three levels because these tasks are feasible simultaneously. The condition of feasibility can be mathematically written as the existence of a set

```
A =
q, ˙q = 0|xdes
i
```

```
.    (57)
```

where **_x_**<sup>des</sup> _i_ is the corresponding desired task value of the task variable **_x_** _i_ defined in (1). The fourth task, however, cannot be accomplished completely because no respective set exists which additionally fulfills **_x_**<sup>des</sup> 4 = **_f_** 4( **_q_** ). Therefore, the completion of this task is dropped due to its minor role in the priority order, but it is executed as good as possible in a locally optimal sense according to the remaining available null space.

It is noticeable that the steady state is reached considerably later in case of the static null space projections. Due

<!-- Start of picture text -->
0.1 0.16<br>Level 1 Level 2<br>0.12<br>0<br>0.08<br>0.04<br>-0.1<br>0<br>-0.2 -0.04<br>0 0.5 1 1.5 0 0.5 1 1.5<br>Time [s] Time [s]<br>1.1<br>0.2 Level 3 Level 4<br>1.0<br>0.1<br>0.9<br>0<br>-0.1 0.8<br>0 0.5 1 1.5 0 0.5 1 1.5<br>Time [s] Time [s]<br>successive, statically consistent, W=I augmented, dynamically consistent, W=M (Sec. 3.3.1)<br>successive, dynamically consistent, W=M augmented, dynamically consistent, W=I (Sec. 3.3.2)<br>augmented, statically consistent, W=I no null space projection<br>-direction [m]Error TCP x -direction [m]Error TCP y<br>Error TCP rotation [rad]<br>Euclidean error norm joint space [-]<br><!-- End of picture text -->

Figure 2: Simulation of different torque control null space projections on a four DOF manipulator with four hierarchy levels

to the dynamic coupling of the tasks, disturbing accelerations are generated across the priority levels and slow down the transient behavior. The reason for that is the existence of inertia couplings between the priority levels. Dynamically consistent null space projectors fulfilling Definition 2 implicitly annihilate these inertia couplings so that the tasks can converge undisturbed. The successive, dynamically consistent solution shows excellent performance on the first priority level, but on the lower levels, the priority order is not strictly ensured, neither dynamically nor statically. On the third level, the steady-state error is even larger than the one in case of simply adding up all control torques without applying any null space projections at all. Considering the two dynamically consistent, augmented projections one can say that they both feature the best performance, but the results are not identical. The final configuration is different which can be clearly seen in the different level four Euclidean error norms in the steady state.

The second simulative study illustrates the benefits of the stiffness consistent null space projection. The slightly modified model in Fig. 3 is used. Four adaptive mechanical springs are placed in between the links. That way, a desired TCP position (in _x_ and _y_ direction) on priority level one can be statically maintained without any power consumption.

<!-- Start of picture text -->
50 Nm rad 50 Nm rad<br>Approaching obstacle 100 Nm rad Level 1<br>(repulsion on Level 2) y<br>z x<br>g<br>100 Nm rad<br><!-- End of picture text -->

Figure 3: Simulation model of a planar, four DOF system. The links are connected via revolute joints. Each link is modeled by a point mass of 1 kg that is placed in the middle of a bar with length 0.5 m. The dynamics are simulated using _g_ = 9 _._ 81 m _/_ s<sup>2</sup> . As depicted, four mechanical springs are placed in between the links. These allow to maintain a TCP position without active control and power consumption. Additional joint damping is introduced with _di_ = 15 Nms _/_ rad for _i_ = 1 _,_ 2 _,_ 3 _,_ 4 so that no DOF are undamped.

<!-- Start of picture text -->
0.15<br>Level 1<br>0.1<br>0.05<br>0<br>-0.05<br>-0.1<br>0 0.5 1 1.5 2 2.5 3 3.5 4<br>Time [s]<br>0.1<br>Level 1<br>0.05<br>0<br>-0.05<br>-0.1<br>-0.15<br>-0.2<br>-0.25<br>-0.3<br>0 0.5 1 1.5 2 2.5 3 3.5 4<br>Time [s]<br>-0.15<br>Level 2<br>-0.2<br>-0.25<br>-0.3<br>Reference<br>-0.35<br>0 0.5 1 1.5 2 2.5 3 3.5 4<br>Time [s]<br>stiffness consistent (W=K), successive/augmented<br>statically consistent (W=I), successive/augmented<br>dynamically consistent (W=M), successive/augmented<br>dynamically consistent (W=I), successive/augmented<br>no null space projection<br>Cartesian error at TCP in  [m]x<br>Cartesian error at TCP in  [m]y<br>Joint value (first joint) [rad]<br><!-- End of picture text -->

Figure 4: Comparative simulations to show the benefits of a stiffness consistent null space projection

Hence the main task control can be deactivated ( **_τ_** 1 = **0** ) due to this task being completely executed by the springs. In the following scenario, the TCP starts at its desired position and an obstacle is approaching the first link of the manipulator as shown in Fig. 3. At _t_ = 0 _._ 5 s, repulsion of the first link is activated (Level two task) with a stiffness of 200 Nm/rad and damping of 10 Nms/rad. Moreover, an additional integral term is used in the control law on level two with gain 10 Nm _/_ (rad s) such that no steady-state error results. That way one can better compare the behavior of all projectors for the same null space control quality (i. e. no steady-state error on level two after the transient). In the upper two diagrams in Fig. 4, the Cartesian errors at the TCP are depicted. As shown in Section 3.4, a stiffness consistent null space projection minimizes the main task level error in a static sense. The plots reflect these theoretical results. Using **_W_** = **_K_** ( **_q_** 0) a small noteworthy error can be observed during the transient, which is reasonable since this null space projector is of static nature only. Although featuring the best performance by far, the stiffness consistent approach also shows a small steady-state error. This is due to the change in the gravity torques because of the large motion in the null space, cf. (54). However, this small error could be easily treated by slight active control in the Cartesian space of the TCP. On a real robot, one would certainly activate such an additional control on the first priority level to compensate for any disturbances and model uncertainties but still let the springs do most of the work.

It is striking that the dynamically consistent projectors perform very poorly during the transient although they use knowledge of the dynamic capabilities of the system by applying the inertia matrix for the null space determination. But the missing knowledge about the additional springs even leads to worse results than the pure statically consistent projector with **_W_** = **_I_** . Summarized, the comparison with the other null space projectors clearly reveals the advantages of the new concept of stiffness consistent projectors for this subclass of robots. Note that due to the use of only two priority levels, there is no difference between successive and augmented null space projections. In the bottom chart in Fig. 4, the joint value of the first joint is depicted as well as the reference value for the respective secondary task collision avoidance. Fig. 5 shows all joint torques. One can easily see that the information contained in **_K_** ( **_q_** 0) leads to completely different control inputs, and the final steady state is reached considerably faster compared to the other approaches. Solely the statically consistent null space projector with **_W_** = **_I_** also converges very fast. This is due to the fact that the stiffness matrix in this simulation example is of diagonal shape and thus closer to the identity matrix than the weighting matrices in the other approaches.

<!-- Start of picture text -->
30 10<br>20 0<br>10 -10<br>0 -20<br>-10 -30<br>0 1 2 3 4 0 1 2 3 4<br>Time [s] Time [s]<br>15 2<br>10<br>0<br>5<br>-2<br>0<br>-5 -4<br>0 1 2 3 4 0 1 2 3 4<br>Time [s] Time [s]<br>stiffness consistent (W=K), successive/augmented<br>statically consistent (W=I), successive/augmented<br>dynamically consistent (W=M), successive/augmented<br>dynamically consistent (W=I), successive/augmented<br>no null space projection<br>Control torque (first joint) [Nm]<br>Control torque (second joint) [Nm]<br>Control torque (third joint) [Nm] Control torque (fourth joint) [Nm]<br><!-- End of picture text -->

Figure 5: Control torques in the four joints of the simulation model

### 4.2 Experiments
In the following experiments, the null space projectors are applied on a real torque controlled robot, namely a DLRKUKA lightweight robot III [Hirzinger et al., 2002] with seven DOF. The task hierarchy is designed as follows:

1. Level ( _m_ 1 = 3): translational Cartesian impedance at the TCP in _x_ -, _y_ -, _z_ -direction to _keep the initial Cartesian position in space_ ( _x_ : forward/backward, _y_ : left/right, _z_ : up/down),

2. Level ( _m_ 2 = 3): Cartesian impedance for the orientation of the TCP about the three axes with _commanded trajectory_ ,

3. Level ( _m_ 3 = 7): complete joint impedance to _maintain the initial joint configuration_ .

The controller gains are given in Table 1. From an initial configuration of the manipulator, a fast trajectory on the second priority level is applied. Within less than 0.7 s, the TCP orientation is commanded to an intermediate state. After a short rest, it is commanded back to the initial state.

The trajectory for the rotation is specified such that its realization requires large motions in the joints of the manipulator. That allows to evaluate different fundamental aspects in one experiment:

- To which extent is the main task on level one disturbed by control actions on level two and three?

- How well is the task on level two executed due to the restrictions imposed by the task on level one?

- How well is the task on level three executed since it conflicts with the task on level two?

The performance of the null space projectors can be compared on the basis of Fig. 6. The first issue to notice is the clear instability of the augmented, dynamically consistent null space projector with **_W_** = **_I_** from Section 3.3.2. At _t ≈_ 1 _._ 2 s, the emergency stop is used. Although this null space projector has the theoretical advantages shown before, it destabilizes the system. Indeed, that is caused by the procedure _torque → acceleration → null space projection → torque_ described in Section 3.3.2. If **_M_** has a very small eigenvalue, then **_M_**<sup>_−_1</sup> will have a very large one, i. e. its inverse. If the current torque to be projected has a contribution in the direction of the corresponding eigenvector, then the acceleration vector will be “aggressively” scaled. In the second step, the null space projection is performed in the acceleration domain. Note that this projection does not use any knowledge about **_M_** since **_W_** = **_I_** . In other words, the acceleration vector is projected and the resulting acceleration points into another direction while still suffering from the scaling performed in the first step. In the third step, one goes back to joint torques, but the previous scaling is not reversed. Summarized, one can say that this null space projector “aggressively” scales a torque, depending on the actual joint configuration and the eigenvalues of **_M_** ( **_q_** ), respectively. The infeasibility of the obtained, projected joint torques then destabilizes the system due to actuator limitations, saturation, and the limited torque control bandwidth. This aspect of instability will be picked up and analyzed further in the discussion in Section 5.

The upper three diagrams on the left side depict the Cartesian position of the TCP and its reference value. Except for the unstable solution and the summed up control actions (“no null space projection”), the main task is statically achieved. Nevertheless, deviations of several centimeters occur during the transient. Against the expectation of superiority based on the theoretical properties, the projectors using the inertia matrix ( **_W_** = **_M_** ) do not perform better than the projectors without use of it ( **_W_** = **_I_** ). On the contrary, they generate larger errors in fact. That can be seen in the _x_ - and _z_ -direction at _t ≈_ 2 s.

As one would expect, the performance on the second level (right column diagrams in Fig. 6) is restricted due to the

<!-- Start of picture text -->
1.6<br>Level 1 Level 2<br>0.40<br>1.4<br>0.38<br>1.2<br>0.36<br>0.34 1.0<br>Emergency stop<br>0.32 0.8<br>0 0.5 1 1.5 2 2.5 3 0 0.5 1 1.5 2 2.5 3<br>Time [s] Time [s]<br>0.1<br>Level 1 Level 2<br>-0.34<br>0<br>-0.36<br>-0.1<br>-0.38<br>-0.2<br>-0.40<br>-0.3<br>-0.42<br>-0.4<br>-0.44<br>0 0.5 1 1.5 2 2.5 3 0 0.5 1 1.5 2 2.5 3<br>Time [s] Time [s]<br>0.31 2.6<br>Level 1 Level 2<br>2.4<br>0.30<br>2.2<br>0.29<br>2.0<br>0.28 1.8<br>1.6<br>0.27<br>0 0.5 1 1.5 2 2.5 3 0 0.5 1 1.5 2 2.5 3<br>Time [s] Time [s]<br>2.0<br>Level 3 reference value<br>successive, statically consistent, W=I<br>1.6 successive, dynamically consistent, W=M<br>augmented, statically consistent, W=I<br>1.2 augmented, dynamically consistent, W=M (Sec. 3.3.1)<br>augmented, dynamically consistent, W=I (Sec. 3.3.2)<br>0.8 with emergency stop at t≈ 1.2s  (instability!)<br>no null space projection<br>0.4<br>0<br>0 0.5 1 1.5 2 2.5 3<br>Time [s]<br>Position of TCP in  [m]x<br>Orientation of TCP about -axis [rad] x<br>Position of TCP in  [m]y<br>Orientation of TCP about -axis [rad] y<br>Position of TCP in  [m]z<br>Orientation of TCP about -axis [rad] z<br>Euclidean norm of joint errors [-]<br><!-- End of picture text -->

Figure 6: Experimental comparison between different torque control null space projections on a seven DOF robot with three priority levels: The first priority level is described by a translational Cartesian impedance in _x_ -, _y_ -, _z_ -direction to keep the initial Cartesian position in space ( _x_ : forward/backward, _y_ : left/right, _z_ : up/down). The second priority level is defined as a Cartesian impedance for the orientation of the TCP about the three axes with commanded trajectory. The third priority level is described by a complete joint impedance to maintain the initial joint configuration.

projection in the null space of the main task. That can be seen in the transient behavior of all three control variables when the desired orientation of the TCP is changed. If the rotational Cartesian impedance was placed on the first priority level instead, then the control errors and the overshootings would be smaller for the given parameterization. Furthermore, the plots on the right confirm the theoretical properties of successive null space projections. As in the simulations, they perform worse than the augmented ones due to the non-strict hierarchy they generate. Therefore, the third priority level interferes with the second level task and leads to large control errors on level two. That effect can be clearly seen in the rotation about the _x_ -axis and _z_ -axis. But the most remarkable result is, that a strict hierarchy (i. e. augmented) does not necessarily require dynamic consistency for high performance _during the transient_ . The comparable performance of the “augmented, statically consistent, **_W_** = **_I_** ” solution and the “augmented, dynamically consistent, **_W_** = **_M_** ” solution in all three directions (right column diagrams in Fig. 6) is not in accordance with the theory. Yet it confirms our results from [Albu-Sch¨affer et al., 2003], where we concluded that the differences between static and dynamic consistency are significantly smaller than expected when real hardware is considered. That effect can be traced back to modeling uncertainties (inertia matrix, kinematics, friction) and disturbances, among others. Nakanishi _et al._ [Nakanishi et al., 2008] came to similar conclusions while comparing inertiaweighted redundancy resolutions among each other. The authors stated that the requirement of a highly accurate, estimated inertia matrix is difficult to realize.

On the third level, the successive null space projections perform better than the augmented ones, because they do not implement a strict hierarchy. Therefore, the task on the lowest priority level three can be executed using a larger accessible workspace. The stable, augmented solutions ( **_W_** = **_I_** , **_W_** = **_M_** ) have a comparable behavior. They establish a strict hierarchy, which implies that the task performance on level three will suffer from the limited available workspace. Therefore, it is proper that the largest error norms will be generated with augmented null space projections. Thanks to the different weighting matrices, the steady-state joint configurations are slightly differing as it can be observed at _t_ = 1 _._ 5 s. Nevertheless, since the actual inertia has no effect in any static configuration, one cannot generalize superiority or inferiority of inertia-based null space projections compared to non-inertia-based solutions in these states.

<!-- Start of picture text -->
Level 1<br>0.6<br>0.4<br>0.2<br>0<br>0 0.5 1 1.5 2 2.5 3<br>Time [s]<br>Level 2<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>0 0.5 1 1.5 2 2.5 3<br>Time [s]<br>reference value<br>successive, statically consistent, W=I<br>successive, dynamically consistent, W=M<br>augmented, statically consistent, W=I<br>augmented, dynamically consistent, W=M (Sec. 3.3.1)<br>augmented, dynamically consistent, W=I (Sec. 3.3.2)<br>with emergency stop at t≈ 1.2s  (instability!)<br>no null space projection<br>Absolute translational error at TCP [m]<br>Absolute orientation error at TCP [rad]<br><!-- End of picture text -->

Figure 7: Absolute errors on the first and second priority level during the experiments

The total errors in the TCP position and the TCP orientation are plotted in Fig. 7. Note that the implemented torque-based tasks realize mechanical impedances. In order to provide the desired physical compliance, the controllers have been implemented following the classical con-

cepts of impedance control [Hogan, 1985], i. e. using PDcontrol laws. For that reason, small steady-state errors occur. By adding an integral component to the control law, one would erase that error. However, the desired massspring-damper behavior, which is beneficial for compliant physical contacts and interaction of the robot with its environment, would be lost then.

## 5 Discussion and Comparison
The main aspects of the following detailed discussion and comparison are summarized in Table 2.

### 5.1 Comparison of Successive and Augmented Null Space Projections
The successive null space projection is computationally efficient due to the decoupled calculations of **_N_**<sup>suc</sup> _i_ ( **_q_** ). However, a projection into the null spaces of all higher priority tasks via (6) does not imply strict compliance with the priority order because the tasks are not orthogonal. The matrix **_N_**<sup>suc</sup> _i_ ( **_q_** ) _∀ i >_ 2 is not idempotent in general, i. e. the projection property is not fulfilled due to **_N_**<sup>suc</sup> _i_ ( **_q_** ) _̸_ = **_N_**<sup>suc</sup> _i_ ( **_q_** ) **_N_**<sup>suc</sup> _i_ ( **_q_** ), which is a well-known drawback. The effect on the implementation results can be interpreted easily: A task torque originating from level _i_ is successively multiplied by _i −_ 1 matrices according to recursion (6). Each multiplication ensures orthogonality to the corresponding higher level task but it also corrupts all preceding projections at the same time, thus the task hierarchy is not strict. Yet the less complex structure of (6) makes it easier to implement dynamic hierarchies such as [Dietrich et al., 2012b], where the priority order can be modified online or tasks get activated and deactivated during operation. The main advantage of the successive projection is that _algorithmic singularities_ are avoided. These arise when tasks on different priority levels conflict with each other. In the augmented projection such a singularity arises when a rank loss occurs in (10). Singularities in **_J_**<sup>aug</sup> _i−_ 1<sup>(</sup><sup>**_q_**)havetobe</sup> avoided by smart choice of the task definitions or treated by applying singularity-robust techniques such as damped least-squares methods [Deo and Walker, 1995]. Hence the use of the method complicates the hierarchy design. But the augmented projection enforces orthogonality of all involved tasks, the projection matrix **_N_**<sup>aug</sup> _i_ ( **_q_** ) always fulfills the idempotence criterion **_N_**<sup>aug</sup> _i_ ( **_q_** ) = **_N_**<sup>aug</sup> _i_ ( **_q_** ) **_N_**<sup>aug</sup> _i_ ( **_q_** ), thus a strict hierarchy is ensured. In fact, a stability proof for a generic hierarchy is only known with augmented projections so far [Nakanishi et al., 2008; Dietrich et al., 2013].

```
i
(q) = N aug
i
(q)N aug
i
(q),
```

```
N suc
i
(q)̸ = N suc
i
(q)N suc
i
```

In successive projections the choice of the weighting matrix cannot solve this problem of a non-strict hierarchy. One has to keep in mind that the type of strictness (successive,

augmented) and the kind of consistency (statically, dynamically, stiffness) are not directly related. Thus a drawback through the choice in the consistency or the strictness cannot be cleared by the choice in the other category. The strictness of the hierarchy determines whether the tasks are properly decoupled or not, and the consistency determines in which way this decoupling is performed, i. e. statically, dynamically or stiffness-related.

For inverse kinematics, a stability analysis as well as a detailed discussion and comparison of the successive and the augmented projection was presented by Antonelli [Antonelli, 2009].

### 5.2 Comparison of Static, Dynamic, and Stiffness Consistency
The consistency is a less clear aspect in contrast to strictness. Although dynamically consistent projections have a clear theoretical advantage due to the dynamical decoupling of the priority levels, the final steady state is also achieved with static consistency. Former comparative simulations [Chang and Khatib, 1995] and the ones in Section 4.1 have revealed that the performance of dynamically consistent projections is superior to the static ones. However, a precise model of the joint inertia matrix is needed. Our experiments on real hardware in Section 4.2 have shown that the differences between the concepts are significantly smaller than expected. These experimental results confirm previous works in the field such as [Albu-Sch¨affer et al., 2003; Nakanishi et al., 2008; Peters et al., 2008]. The difference between theoretical superiority and practice can be traced back to modeling uncertainties (inertia matrix, kinematics, friction) and disturbances, for example. In [Nakanishi et al., 2008] the authors say that all approaches using the inertia matrix _“[...] significantly degrade, especially in the tasks with fast movements. This implies that these algorithms require highly accurate inertia matrix estimation to be successful”_ and they also trace the problems back to inaccuracies of the estimated inertia matrix. In [Peters et al., 2008] different redundancy resolution techniques are compared but all of them exploit the inertia matrix either more or less. The authors draw the conclusion that the more influence the inertia matrix has in the control law, the worse the experimental results are. They also experience that simulated results are significantly better due to the perfectly known inertia matrix. In our previous work [Albu-Sch¨affer et al., 2003] the first experimental comparison between statically consistent and dynamically consistent null space projections has been performed. The results match with the more extensive and detailed experiments performed here.

Formal stability proofs for task hierarchies are quite intricate [Nakanishi et al., 2008] and they are only known for dynamically consistent resolutions so far. In case of two-

Table 2: Comparison of different torque control null space projections. Note that the stiffness consistent null space projector cannot be easily compared to each approach in a fair way since it is only applicable to a specific subclass of robots, where mechanical springs are mounted in parallel to the robot joints.

```
W = I
W = M(q)
W = I
W = M(q)
W = I
```

||**successive,**<br>**stat. cons.**<br>**_W_** =**_I_**|**successive,**<br>**dyn. cons.**<br>**_W_** =**_M_(****_q_)**|**augmented,**<br>**stat. cons.**<br>**_W_** =**_I_**|**augmented,**<br>**dyn. cons.**<br>**_W_** =**_M_**(**_q_**)|**augmented,**<br>**dyn. cons.**<br>**_W_** =**_I_**|**no**<br>**null space**<br>**projection**|
|---|---|---|---|---|---|---|
|strict hierarchy (static)|main task|main task|yes|yes|yes|no|
|strict hierarchy (dynamic)|no|main task|no|yes|yes|no|
|continuous (no task sing.)|no|no|no|no|no|yes|
|continuous (no algorith. sing.)|yes|yes|no|no|no|yes|
|inertia matrix model-free|yes|no|yes|no|no|yes|
|idempotent (**_N_** <sup>2 </sup>=**_N_**)|no|no|yes|yes|yes|no|
|load independence|yes|main task|yes|yes|no|yes|
|stable in experiments|yes|yes|yes|yes|no|yes|

level hierarchies, see [Ott et al., 2008; Platt et al., 2011] for example. A formal stability proof for a hierarchy with an arbitrary number of priority levels can be found in our recent work [Dietrich et al., 2013].

In Section 3.3 we have detailed two different kinds of dynamically consistent hierarchies. The first one in Section 3.3.1 is a generalized version of the well-known projector by Khatib [Khatib, 1987] which uses the inertia matrix as weighting matrix in the pseudoinversion. Indeed, an infinite number of weighting matrices (36) fulfill the same criteria. The second dynamically consistent projector (41), explained in Section 3.3.2, is of static consistency originally since it refers to a null space projection on acceleration level. The solution was then extended to dynamic consistency by taking the inertia matrix into account in a second step. These two different projectors have basically very similar theoretical properties as illustrated in Table 2. However, the beneficial property of load independence cannot be concluded for Section 3.3.2. Furthermore, we encountered severe stability problems during the experiments with (41). In Section 4.2 we have already explained the reason for the instability. The effect is of structural nature and arises from a configuration-dependent scaling from input torque to projected output torque. In configurations where the inertia matrix has one or more small eigenvalues, the null space projection may lead to infeasible joint torques which exceed the actuator limitations and the torque control bandwidth. Nevertheless, one has to remark that this “aggressive” scaling does not necessarily have to happen, since it depends on the condition of the inertia matrix and the torque to be projected. The simulations in Section 4.1 have depicted two scenarios in which the closed loop behaved properly when applying the acceleration-based null space projector. Our conclusion is that (41) is risky to be applied, and since

other null space projectors have additional beneficial properties while not suffering from stability issues, there is no convincing reason for the use of (41).

It shall also be noted that one can easily obtain a dynamically consistent null space projector while completely avoiding any expensive numerical computations such as singular value decompositions. The only adaptation is to further subdivide all levels from (2) such that _mi_ = 1 _∀i_ , which does not pose any problems in general. If a set of equally prioritized tasks is feasible, a strict hierarchy among these subtasks is also feasible. Then the inversion in (13) simplifies to the inversion of a scalar. A formulation with reduced computational complexity is particularly suitable for realtime applications of dynamic hierarchies where subtasks are activated and deactivated online and the priority order is modified during operation, e. g. by utilizing physically interpretable measures as done in [Dietrich et al., 2012b].

Stiffness consistency (Section 3.4) can be interpreted as a subclass of static consistency with particular properties for specific scenarios. In Section 4.1 we have demonstrated the advantages of this new null space projector in simulation. In case of mechanical springs placed in parallel to the joints, a main task can be statically achieved by these passive elements without any power consumption or active control. By applying the stiffness consistent null space projector, the main task execution through the springs can be kept undisturbed while a secondary task is executed in its null space. For such a scenario, the stiffness consistent resolution is superior to any other null space projection.

## 6 Conclusion
An overview of established torque control null space projections was given. The discussion comprised the strictness

of the control task hierarchy by comparing successive and augmented techniques. The second main aspect treated the consistency of the projections, i. e. static, dynamic, and the novel idea of stiffness consistency. The latter allows to project subtasks into the null space of higher priority tasks which are executed by mechanical springs. Knowledge about these spring elements is used in the projector computation. Moreover, we have generalized the popular dynamically consistent projector by Khatib [Khatib, 1987] and interpreted his weighting matrix in the pseudoinversion as an intuitive special case of an infinite number of weighting matrices. Furthermore, another type of dynamically consistent projectors has been analyzed which originates from an acceleration-based approach but can be extended to torque control. Extensive simulations and experiments illustrated the differences in all null space projections from a theoretical and practical point of view. A thorough discussion and comparison of the approaches concluded this survey.

## Acknowledgements
We would like to thank Pierre-Brice Wieber for his helpful comments on the dynamically consistent null space projector in Section 3.3.2.

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Appendix
### Dynamic Consistency of (36)
The proof for dynamic consistency of (36) according to Definition 2 is provided in the following. For the sake of simplicity, the dependencies on **_q_** are omitted. The derivation is based on the well-known relationships [Khatib, 1987; Dietrich et al., 2013]

```
N 2 = I −JT
1 (JW +
1
)T    (58)
I −JT
1 (JW +
1
)T = I −XT
1 (XW +
1
)T    (59)
I −XT
1 (XW +
1
)T = W T Y T
1 (Y 1W T Y T
1 )−1Y 1    (60)
```

Eq. (59) describes the invariance of the projector calculcation to the singular values of the Jacobian matrix which cancel out. Eq. (60) states the equality of “substracting” the range space of **_J_** 1 from the unconstrained space **_I_** and obtaining the null space projector by directly using **_Y_** . Now the _dynamic consistency_ of

```
N 2 = W T Y T
1 (Y 1W T Y T
1 )−1Y 1 ,    (61)
W = XT
1 X1BX + BY Y T
1 Y 1M    (62)
```

can be shown as follows. Since **_X_** 1 **_Y_**<sup>_T_</sup> 1<sup>=</sup><sup>**0**</sup>, the sim pl i fica- tion

```
W T Y T
1 = (BT
XXT
1 X1 + MY T
1 Y 1BT
Y )Y T
1
= MY T
1 Y 1BT
Y Y T
1    (63)
```

can be made. Definition 2 with (27) is fulfilled:

```
J1M −1N 2 = J1M −1MY T
1 Y 1BT
Y Y T
1 (Y 1W T Y T
1 )−1Y 1
= USV T Y T
1 Y 1BT
Y Y T
1 (Y 1W T Y T
1 )−1Y 1
= 0 .    (64)
```

Eq. (64) can be concluded because

```
V T Y T
1 =
 X1
Y 1
Y T
1 =
 0
I
,
```

and

```
0
I
= 0 .
```

If the formulation on the right of (60) is used, the conditions on the rank of **_BX_** can even be dropped. In contrast to the formulations in (59), the algorithm does not use **_W_**<sup>_−_1</sup> so that

```
(65)
```

replaces the necessary (but not sufficient) condition (37) for the inversion ( **_Y_** 1 **_W_**<sup>_T_</sup> **_Y_**<sup>_T_</sup> 1<sup>)</sup><sup>_−_1.</sup>

### Dynamic Consistency of (41)
The proof for dynamic consistency of (41) according to Definition 2 is provided in the following. For the sake of simplicity, the dependencies on **_q_** are omitted.

```
J1M −1N 2 = J1M −1M(I −JW +
1
J1)M −1
= (J1 −J1W −1JT
1 (J1W −1JT
1 )−1
|
{z
}
```

```
J1)M −1
= 0 .
```

## References
- Albu-Sch¨affer, A., Ott, C., Frese, U., and Hirzinger, G. (2003). Cartesian Impedance Control of Redundant Robots: Recent Results with the DLR-Light-WeightArms. In _Proc. of the 2003 IEEE International Conference on Robotics and Automation_ , pages 3704–3709.

```
USV T Y T
1 = US
 0
I
```

- Antonelli, G. (2009). Stability Analysis for Prioritized Closed-Loop Inverse Kinematic Algorithms for Redundant Robotic Systems. _IEEE Transactions on Robotics_ , 25(5):985–994.

- Antonelli, G., Indiveri, G., and Chiaverini, S. (2009). Prioritized Closed-Loop Inverse Kinematic Algorithms for Redundant Robotic Systems with Velocity Saturations. In _Proc. of the 2009 IEEE/RSJ International Conference on Intelligent Robots and Systems_ , pages 5892–5897.

- Au, S. K. and Herr, H. M. (2009). Powered Ankle-Foot Prosthesis. _IEEE Robotics & Automation Magazine_ , 15(3):52–59.

- Baerlocher, P. and Boulic, R. (1998). Task-Priority Formulations for the Kinematic Control of Highly Redundant Articulated Structures. In _Proc. of the 1998 IEEE/RSJ International Conference on Intelligent Robots and Systems_ , pages 323–329.

- Baerlocher, P. and Boulic, R. (2004). An Inverse Kinematic Architecture Enforcing an Arbitrary Number of Strict Priority Levels. _The Visual Computer_ , 20(6):402–417.

- Baillieul, J., Hollerbach, J. M., and Brockett, R. (1984). Programming and control of kinematically redundant manipulators. In _Proc. of the 23rd IEEE Conference on Decision and Control_ , pages 768–774.

- Chang, K.-S. and Khatib, O. (1995). Manipulator Control at Kinematic Singularities: A Dynamically Consistent Strategy. In _Proc. of the 1995 IEEE/RSJ International Conference on Intelligent Robots and Systems_ , pages 84– 88.

- Chiaverini, S. (1997). Singularity-Robust Task-Priority Redundancy Resolution for Real-Time Kinematic Control of Robot Manipulators. _IEEE Transactions on Robotics and Automation_ , 13(3):398–410.

- Decr´e, W., Smits, R., Bruyninckx, H., and De Schutter, J. (2009). Extending iTaSC to support inequality constraints and non-instantaneous task specification. In _Proc. of the 2009 IEEE International Conference on Robotics and Automation_ , pages 964–971.

- Deo, A. and Walker, I. (1995). Overview of Damped LeastSquares Methods for Inverse Kinematics of Robot Manipulators. _Journal of Intelligent Robotic Systems_ , 14(1):43– 68.

- Dietrich, A., Albu-Sch¨affer, A., and Hirzinger, G. (2012a). On Continuous Null Space Projections for Torque-Based, Hierarchical, Multi-Objective Manipulation. In _Proc. of the 2012 IEEE International Conference on Robotics and Automation_ , pages 2978–2985.

- Dietrich, A., Ott, C., and Albu-Sch¨affer, A. (2013). MultiObjective Compliance Control of Redundant Manipulators: Hierarchy, Control, and Stability. In _Proc. of the 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems_ , pages 3043–3050.

- Dietrich, A., Wimb¨ock, T., Albu-Sch¨affer, A., and Hirzinger, G. (2012b). Integration of Reactive, TorqueBased Self-Collision Avoidance Into a Task Hierarchy. _IEEE Transactions on Robotics_ , 28(6):1278–1293.

- Dietrich, A., Wimb¨ock, T., Albu-Sch¨affer, A., and Hirzinger, G. (2012c). Reactive Whole-Body Control: Dynamic Mobile Manipulation Using a Large Number of Actuated Degrees of Freedom. _IEEE Robotics & Automation Magazine_ , 19(2):20–33.

- Dollar, A. M. and Herr, H. (2008). Lower Extremity Exoskeletons and Active Orthoses: Challenges and State-ofthe-Art. _IEEE Transactions on Robotics_ , 24(1):144–158.

- Doty, K. L., Melchiorri, C., and Bonivento, C. (1993). A Theory of Generalized Inverses Applied to Robotics. _International Journal of Robotics Research_ , 12(1):1–19.

- Featherstone, R. (2010). Exploiting Sparsity in Operational-space Dynamics. _International Journal of Robotics Research_ , 29(10):1353–1368.

- Featherstone, R. and Khatib, O. (1997). Load Independence of the Dynamically Consistent Inverse of the Jacobian Matrix. _International Journal of Robotics Research_ , 16(2):168–170.

- Grimmer, M., Eslamy, M., Gliech, S., and Seyfarth, A. (2012). A Comparison of Parallel- and Series Elastic Elements in an actuator for Mimicking Human Ankle Joint in Walking and Running. In _Proc. of the 2012 IEEE International Conference on Robotics and Automation_ , pages 2463–2470.

- Haeufle, D. F. B., Taylor, M. D., Schmitt, S., and Geyer, H. (2012). A clutched parallel elastic actuator concept: towards energy efficient powered legs in prosthetics and robotics. In _Proc. of The Fourth IEEE/RAS/EMBS International Conference on Biomedical Robotics and Biomechatronics_ , pages 1614–1619.

- Hirzinger, G., Sporer, N., Albu-Sch¨affer, A., H¨ahnle, M., Krenn, R., Pascucci, A., and Schedl, M. (2002). DLR’s torque-controlled light weight robot III - are we reaching the technological limits now? In _Proc. of the 2002 IEEE International Conference on Robotics and Automation_ , pages 1710–1716.

- Hogan, N. (1985). Impedance Control: An Approach to Manipulation: Part I - Theory, Part II - Implementation,

Part III - Applications. _Journal of Dynamic Systems, Measurement, and Control_ , 107:1–24.

- Hollerbach, J. H. and Suh, K. C. (1987). Redundancy Resolution of Manipulators through Torque Optimization. _IEEE Journal of Robotics and Automation_ , RA-3(4):308– 316.

- Kanoun, O., Lamiraux, F., and Wieber, P.-B. (2011). Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task. _IEEE Transactions on Robotics_ , 27(4):785–792.

- Khatib, O. (1987). A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation. _IEEE Journal of Robotics and Automation_ , RA-3(1):43–53.

- Khatib, O. (1995). Inertial Properties in Robotic Manipulation: An Object-Level Framework. _International Journal of Robotics Research_ , 14(1):19–36.

- Khatib, O., Sentis, L., Park, J., and Warren, J. (2004). Whole-Body Dynamic Behavior and Control of Humanlike Robots. _International Journal of Humanoid Robots_ , 1(1):29–43.

- Lee, J., Mansard, N., and Park, J. (2011). Intermediate Desired Value Approach for Continuous Transition among Multiple Tasks of Robots. In _Proc. of the 2011 IEEE International Conference on Robotics and Automation_ , pages 1276–1282.

- Maciejewski, A. A. and Klein, C. A. (1989). The Singular Value Decomposition: Computation and Application to Robotics. _International Journal of Robotics Research_ , 8(6):63–79.

- Mansard, N., Khatib, O., and Kheddar, A. (2009). A Unified Approach to Integrate Unilateral Constraints in the Stack of Tasks. _IEEE Transactions on Robotics_ , 25(3):670–685.

- Nakamura, Y., Hanafusa, H., and Yoshikawa, T. (1987). Task-Priority Based Redundancy Control of Robot Manipulators. _International Journal of Robotics Research_ , 6(2):3–15.

- Nakanishi, J., Cory, R., Mistry, M., Peters, J., and Schaal, S. (2008). Operational Space Control: A Theoretical and

Empirical Comparison. _International Journal of Robotics Research_ , 27(6):737–757.

- Ott, C., Kugi, A., and Nakamura, Y. (2008). Resolving the Problem of Non-integrability of Nullspace Velocities for Compliance Control of Redundant Manipulators by using Semi-definite Lyapunov functions. In _Proc. of the 2008 IEEE International Conference on Robotics and Automation_ , pages 1999–2004.

- Park, J. (1999). _Analysis and Control of Kinematically Redundant Manipulators: An Approach Based on Kinematically Decoupled Joint Space Decomposition_ . PhD thesis, Pohang University of Science and Technology.

- Peters, J., Mistry, M., Udwadia, F., Nakanishi, J., and Schaal, S. (2008). A unifying framework for robot control with redundant DOFs. _Autonomous Robots_ , 24(1):1–12.

- Platt, R., Abdallah, M., and Wampler, C. (2011). Multiplepriority impedance control. In _Proc. of the 2011 IEEE International Conference on Robotics and Automation_ , pages 6033–6038.

- Sadeghian, H., Villani, L., Keshmiri, M., and Siciliano, B. (2013). Dynamic multi-priority control in redundant robotic systems. _Robotica_ , 31(7):1155–1167.

- Sentis, L. and Khatib, O. (2005). Synthesis of Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives. _International Journal of Humanoid Robotics_ , 2(4):505–518.

- Siciliano, B. and Slotine, J.-J. (1991). A General Framework for Managing Multiple Tasks in Highly Redundant Robotic Systems. In _Proc. of the 5th International Conference on Advanced Robotics_ , pages 1211–1216.

- Sugiura, H., Gienger, M., Janssen, H., and Goerick, C. (2010). Reactive Self Collision Avoidance with Dynamic Task Prioritization for Humanoid Robots. _International Journal of Humanoid Robots_ , 7(1):31–54.

- Winfree, K. N., Stegall, P., and Agrawal, S. K. (2011). Design of a Minimally Constraining, Passively Supported Gait Training Exoskeleton: ALEX II. In _Proc. of the 2011 IEEE International Conference on Rehabilitation Robotics_ , pages 1–6.
