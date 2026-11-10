"""
Data quality module.
"""


class DataQuality:
    def __init__(self, active=True):
        self._active = active

    def plot(self, filename, obj, **kwargs):

        #        if plot_type == "scanblock":
        #            ifnum = obj[0].ifnum
        #            plnum = obj[0].plnum
        #            #fdnum = obj[0].fdnum
        #            projid = obj[0].meta[0]["PROJID"]
        #            source = obj[0].meta[0]["OBJECT"]
        #
        #        fnm = outdir / filename

        if self._active:
            p = obj.plot(**kwargs)
            p.savefig(filename, hidebuttons=False, dpi=150)
