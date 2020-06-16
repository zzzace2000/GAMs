from .base import MyGAMPlotMixinBase
from .EncodingBase import LabelEncodingRegressorMixin, LabelEncodingClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class MySKLearnGBTMixin(MyGAMPlotMixinBase):
    def __init__(self, n_estimators=20000, max_depth=1, learning_rate=None, n_iter_no_change=50,
        validation_fraction=0.176, tol=0., random_state=1377, subsample=1.0, min_samples_split=2,
        min_samples_leaf=1, max_features=None, **kwargs):

        if learning_rate is None:
            learning_rate = 0.1 if isinstance(self, GradientBoostingClassifier) else 1.0

        super().__init__(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, n_iter_no_change=n_iter_no_change, validation_fraction=validation_fraction,
            tol=tol, subsample=subsample, min_samples_split=min_samples_split, max_features=max_features,
            min_samples_leaf=min_samples_leaf, **kwargs)

    def get_GAM_plot_dataframe(self, x_values_lookup=None, center=True):
        assert self.is_GAM, 'Only supports visualization of max_depth=1 but with depth %d' % self.max_depth
        return super().get_GAM_plot_dataframe(x_values_lookup=x_values_lookup, center=center)

    @property
    def is_GAM(self):
        return self.max_depth == 1

    @property
    def param_distributions(self):
        return {
            'learning_rate': [0.5, 0.1, 0.05, 0.01],
            'subsample': [1., 0.9, 0.8, 0.6],
            'max_features': [1., 0.9, 0.8, 0.6],
        }


class MySKLearnGBTClassifier(LabelEncodingClassifierMixin, MySKLearnGBTMixin, GradientBoostingClassifier):
    pass

class MySKLearnGBTRegressor(LabelEncodingRegressorMixin, MySKLearnGBTMixin, GradientBoostingRegressor):
    pass
