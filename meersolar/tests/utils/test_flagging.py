import pytest
import traceback
import os
from casatasks import casalog
from casatools import table
from unittest.mock import patch, MagicMock, call
from meersolar.utils.flagging import *

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    traceback.print_exc()
    pass


def test_do_flag_backup(dummy_msname):
    from casatasks import flagmanager

    do_flag_backup(dummy_msname, flagtype="test_flagdata")
    flags = flagmanager(vis=dummy_msname, mode="list")
    flagged = False
    for f in flags:
        if f != "MS":
            ver_name = flags[f]["name"]
            if "test_flagdata" in ver_name:
                flagmanager(vis=dummy_msname, mode="delete", versionname=ver_name)
                if flagged is not True:
                    flagged = True
    assert flagged == True


def test_get_unflagged_antennas(dummy_submsname):
    tb = table()
    tb.open(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", nomodify=False)
    flag = tb.getcol("FLAG")
    flag *= False
    tb.putcol("FLAG", flag)
    tb.flush()
    tb.close()
    antlist, fraclist = get_unflagged_antennas(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms"
    )
    antlist = antlist.tolist()
    fraclist = fraclist.tolist()
    assert antlist == [
        "m000",
        "m001",
        "m002",
        "m003",
        "m004",
        "m005",
        "m006",
        "m007",
        "m008",
        "m009",
        "m010",
        "m011",
        "m014",
        "m015",
        "m016",
        "m017",
        "m018",
        "m020",
        "m021",
        "m022",
        "m023",
        "m024",
        "m025",
        "m026",
        "m027",
        "m028",
        "m029",
        "m031",
        "m032",
        "m033",
        "m034",
        "m035",
        "m036",
        "m037",
        "m038",
        "m039",
        "m040",
        "m041",
        "m042",
        "m043",
        "m044",
        "m045",
        "m046",
        "m047",
        "m048",
        "m049",
        "m050",
        "m051",
        "m052",
        "m053",
        "m054",
        "m055",
        "m056",
        "m057",
        "m058",
        "m059",
        "m060",
        "m061",
        "m063",
    ]
    assert fraclist == [0] * len(antlist)


def test_get_chans_flag(dummy_submsname):
    unflag_chans, flag_chans = get_chans_flag(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms"
    )
    assert unflag_chans == [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flag_chans == []


def test_calc_flag_fraction(dummy_submsname):
    frac = calc_flag_fraction(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms")
    assert frac == 0.0


@pytest.mark.parametrize(
    "uvrange", ["100~200", ">100", "<200", "200~300lambda", ">200lambda", "<300lambda"]
)
def test_flag_outside_uvrange(dummy_submsname, uvrange):
    assert (
        flag_outside_uvrange(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", uvrange)
        == 0
    )
    tb = table()
    tb.open(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", nomodify=False)
    flag = tb.getcol("FLAG")
    flag *= False
    tb.putcol("FLAG", flag)
    tb.flush()
    tb.close()
