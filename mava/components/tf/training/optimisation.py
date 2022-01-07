# TODO (Arnu): need to create optimizer and averaging callbacks

# if self._target_averaging:
# assert 0.0 < self._target_update_rate < 1.0
# tau = self._target_update_rate
# for src, dest in zip(online_variables, target_variables):
#     dest.assign(dest * (1.0 - tau) + src * tau)