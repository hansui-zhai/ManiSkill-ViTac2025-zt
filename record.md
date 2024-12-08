# mani_vitac

## Track1

### 文件结构-1

1. asssets: 结构模型文件

2. configs: 各种配置参数文件

3. envs: 仿真的环境生成文件，**其中包含一些训练的默认环境参数，但是在后续训练中被从配置文件中读取的参数覆盖掉一部分**

4. scripts-
arguments: 训练参数替换API
evaluation: 评价上传，与我们无关
universal_training_script: 像一个框架，将各个部分拼接在一起（训练环境、训练参数、训练网络、训练策略）

5. solution-
actor_and_critic: 重写演员（TD3）和评论家（通用）
feature_extractors: 针对不同数据类型的特征提取网络（为Actor做头，提取点流；为Critic做头或者说完全提取网络，提取不同特征），在训练时传递给模型中，包括Actor和Critic都有使用**或者说把输入处理成torch.tensor张量的形式**
networks: 其中定义的网络为进一步特征提取的模块，构成了Actor网络中的一部分
policies: 构建训练策略，主要是重写生成各自两个任务的Actor和Critic，universal_training_script会调用

### 总结TD3各输入输出量的格式-1

#### peg_insertion

为什么这个跑不起来？它大到哪里了？

1. Actor输入：
    obs["marker_flow"]-表面上的点的二维坐标

    ```bash
    [
        [
         [[u,v]_1,[u,v]_2,...,[u,v]_126]_o,
         [[u,v]_1,[u,v]_2,...,[u,v]_126]_c
        ]_r,

        [
         [[u,v]_1,[u,v]_2 ...,[u,v]_126]_o,
         [[u,v]_1,[u,v]_2 ...,[u,v]_126]_c
        ]_l
    ]
    ```

    但是这个uv中有一半多的冗余（2-128-2-2）(二维坐标-63个点+65个冗余点-初始与当前-左右传感器)
    (batch_num, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v)) # 从feature_extractors.py中处理

2. Actor输出：

    ```bash
    # 在环境的PY文件中
    # 没有被配置文件覆盖掉
    self.action_space = spaces.Box(
            low=-1, high=1, shape=max_action.shape, dtype=np.float32
        )
    ```

3. Critic输入：
    评论网络的其中一个输入是 gt_offset，其dim = 3

    定义了两个评论者，而且我没有看出评论者用的什么网络(creatmlp根据要求生成的？MLP)，估计是搞了点小优化，复用了网络target？因为如果展开写的话，TD3的评论家应该有四个才对

    总输入：feature_dim = 3 + action_dim = 3？对的，是一个六维的输入

    ```bash
    qvalue_input = torch.cat([features, actions], dim=1)
    ```

    **creatmlp：创建一个多层感知机 (MLP)，它是一组全连接层，每层后面都有一个激活函数**
    根据参数创建：input_dim，output_dim，net_arch（神经网络的架构它表示每层的单元数。此列表的长度即为层数。）

    ```bash
    # Default network architecture, from the original paper
    # 用的TD3policy默认的网络结构？
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]

    # 或者配置文件中的？
    net_arch:
      pi: [256, 256]
      qf: [256, 256]

    assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
    assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
    assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
    actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    ```

    但是感觉很不合理，一个6维的向量要400-300的网络去感知

4. Critic输出：
   q值-标量？

#### long_open_lock

1. Actor输入：

    ```bash
    # 在actor_and_critic中处理，感觉写的很乱，简直稀烂
    # (batch_num, 2 (left_and_right), 128 (marker_num), 4 (u0, v0; u1, v1)) 这是输入的时候调整之后的形状
    # 但在env中还是（2，2，128，2），两个是一样的
    ```

2. Actor输出：

   ```bash
    # 在环境的PY文件中
    # 没有被配置文件覆盖掉
    self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
   ```

3. Critic输入：
    评论网络的其中一个输入是 key_1,key_2，其dim = 6

    总输入：feature_dim = 6 + action_dim = 2？对的，是一个八维的输入

4. Critic输出：
   q值-标量？

#### 两个任务对比

1. 场景
   两个传感器+一个孔+一个结构件，两者大差不差
2. 配置文件
   **有待对比？**
3. Actor输入
   两个传感器的标记点光流，也没有区别
4. Actor输出
   钉子是一个三维动作向量
   锁孔是一个二维动作向量
5. Critic输入
   钉子

## Track2

### 文件结构-2

1. asssets: 结构模型文件

2. configs: 各种配置参数文件

3. envs: 仿真的环境生成文件，**其中包含一些训练的默认环境参数，但是在后续训练中被从配置文件中读取的参数覆盖掉一部分**

4. scripts-
arguments: 训练参数替换API
evaluation: 评价上传，与我们无关
universal_training_script: 像一个框架，将各个部分拼接在一起（训练环境、训练参数、训练网络、训练策略）

5. solution-
actor_and_critic: 重写演员（TD3）和评论家（通用），**但后续貌似只有评论家那个被调用了**
feature_extractors: 针对不同数据类型的特征提取网络（为Actor做头，提取点流；为Critic做头或者说完全提取网络，提取不同特征），在训练时传递给模型中，包括Actor和Critic都有使用**或者说把输入处理成torch.tensor张量的形式****示例中只用了FeatureExtractorState，其他的要用得自己改**
networks: 其中定义的网络为进一步特征提取的模块，构成了Actor网络中的一部分
policies: 构建训练策略，主要是重写生成各自两个任务的Actor和Critic，universal_training_script会调用，**但是这里只重写了critic，而Actor貌似用的stablebaseline默认的**

### 总结TD3各输入输出量的格式-2

1. Actor输入：
    到底在哪定义的？
    只知道它没有重写make的方法，但不知道默认的方法如何实现输入的
    既然是默认的，那么一定和Actor内部有关：features_dim是如何传入的?

    ```bash

    ```

2. Actor输出：

    ```bash
    # 在环境的PY文件中
    # 没有被配置文件覆盖掉
    assert max_action_mm_deg.shape == (4,)
    self.max_action_mm_deg = max_action_mm_deg
    self.action_space = spaces.Box(
            low=-1, high=1, shape=max_action_mm_deg.shape, dtype=np.float32
        )
    ```

3. Critic输入：

   ```bash
    # 同样是两个维度联合
    self.features_dim + action_dim

    # features_dim=9
    super(FeatureExtractorState, self).__init__(observation_space, features_dim=9)
    
    gt_offset = observations["gt_offset"]  # 4
    relative_motion = observations["relative_motion"]  # 4 
    gt_direction = observations["gt_direction"] # 1
    return torch.cat([gt_offset, relative_motion, gt_direction], dim=-1)
   ```

4. Critic输出：
    q值-标量？
