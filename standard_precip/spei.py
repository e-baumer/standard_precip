from standard_precip.base_sp import BaseStandardIndex

            
class SPEI(BaseStandardIndex):
    
    def set_distribution_params(self, dist_type='fisk', **kwargs):
        super(SPEI, self).set_distribution_params(dist_type=dist_type, **kwargs)

