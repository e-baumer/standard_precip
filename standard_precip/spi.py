from standard_precip.base_sp import BaseStandardIndex


class SPI(BaseStandardIndex):

    def set_distribution_params(self, dist_type='gengamma', **kwargs):
        super(SPI, self).set_distribution_params(dist_type=dist_type, **kwargs)
