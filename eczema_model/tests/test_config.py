from regression_model.config.config import TRAININGDATA, WRONGDATAPATH, DATA


def test_TRAININGDATA():
    
    assert DATA[-3:] == "csv"

def test_WRONGDATAPATH():
    assert WRONGDATAPATH != ""

    