from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import Actor, TD3Policy

from Track_1.solutions.actor_and_critics import CustomCritic, PointNetActor, LongOpenLockPointNetActor
from Track_1.solutions.feature_extractors import (CriticFeatureExtractor,
                                                  FeatureExtractorForPointFlowEnv, CriticFeatureExtractorForLongOpenLock)


class TD3PolicyForPointFlowEnv(TD3Policy):
    def __init__(
            self,
            *args,
            pointnet_in_dim,
            pointnet_out_dim,
            pointnet_batchnorm,
            pointnet_layernorm,
            zero_init_output,
            use_relative_motion: bool,
            **kwargs,
    ):
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_out_dim = pointnet_out_dim
        self.pointnet_layernorm = pointnet_layernorm
        self.pointnet_batchnorm = pointnet_batchnorm
        self.zero_init_output = zero_init_output
        self.use_relative_motion = use_relative_motion
        super(TD3PolicyForPointFlowEnv, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, FeatureExtractorForPointFlowEnv(self.observation_space)
        )

        return PointNetActor(
            pointnet_in_dim=self.pointnet_in_dim,
            pointnet_out_dim=self.pointnet_out_dim,
            batchnorm=self.pointnet_batchnorm,
            layernorm=self.pointnet_layernorm,
            zero_init_output=self.zero_init_output,
            use_relative_motion=self.use_relative_motion,
            **actor_kwargs,
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractor(self.observation_space)
        )

        return CustomCritic(**critic_kwargs).to(self.device)


class TD3PolicyForLongOpenLockPointFlowEnv(TD3Policy):
    def __init__(
            self,
            *args,
            pointnet_in_dim,
            pointnet_out_dim,
            pointnet_batchnorm,
            pointnet_layernorm,
            zero_init_output,
            use_relative_motion: bool,
            **kwargs,
    ):
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_out_dim = pointnet_out_dim
        self.pointnet_layernorm = pointnet_layernorm
        self.pointnet_batchnorm = pointnet_batchnorm
        self.use_relative_motion = use_relative_motion
        self.zero_init_output = zero_init_output
        super(TD3PolicyForLongOpenLockPointFlowEnv, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs,
        )
        return LongOpenLockPointNetActor(
            pointnet_in_dim=self.pointnet_in_dim,
            pointnet_out_dim=self.pointnet_out_dim,
            batchnorm=self.pointnet_batchnorm,
            layernorm=self.pointnet_layernorm,
            zero_init_output=self.zero_init_output,
            use_relative_motion=self.use_relative_motion,
            **actor_kwargs,
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractorForLongOpenLock(self.observation_space)
        )
        return CustomCritic(**critic_kwargs).to(self.device)


"""
    更改使用其他的Policy, 注意SAC与TD3的Actor是不通用的
"""
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.policies import Actor as SACActor
from Track_1.solutions.actor_and_critics import SACPointNetActor

class SACPolicyForPointFlowEnv(SACPolicy):
    def __init__(
            self, 
            *args,
            pointnet_in_dim,
            pointnet_out_dim,
            pointnet_batchnorm,
            pointnet_layernorm,
            zero_init_output,
            use_relative_motion: bool,
            **kwargs
    ):
        """
        *args:
            observation_space, 
            action_space, 
            lr_schedule, 
            net_arch = None, 
            activation_fn = nn.ReLU, 
            use_sde = False, 
            log_std_init = -3, 
            use_expln = False, 
            clip_mean = 2, 
            features_extractor_class = ..., 
            features_extractor_kwargs = None, 
            normalize_images = True, 
            optimizer_class = th.optim.Adam, 
            optimizer_kwargs = None, 
            n_critics = 2, 
            share_features_extractor = False
        **kwargs:
            ?
        """
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_out_dim = pointnet_out_dim
        self.pointnet_layernorm = pointnet_layernorm
        self.pointnet_batchnorm = pointnet_batchnorm
        self.zero_init_output = zero_init_output
        self.use_relative_motion = use_relative_motion
        super(SACPolicyForPointFlowEnv,self).__init__(*args, **kwargs)

    # 重写make_actor
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SACActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, FeatureExtractorForPointFlowEnv(self.observation_space)
        )

        return SACPointNetActor(
            pointnet_in_dim=self.pointnet_in_dim,
            pointnet_out_dim=self.pointnet_out_dim,
            batchnorm=self.pointnet_batchnorm,
            layernorm=self.pointnet_layernorm,
            zero_init_output=self.zero_init_output,
            use_relative_motion=self.use_relative_motion,
            **actor_kwargs,
        ).to(self.device)
    
    # 重写make_critic
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractor(self.observation_space)
        )

        return CustomCritic(**critic_kwargs).to(self.device)