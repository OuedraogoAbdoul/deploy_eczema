from eczema_model.config.config import DATA, TRAININGDATA, WRONGDATAPATH


def test_TRAININGDATA():
    
    assert DATA[-3:] == "csv"

def test_WRONGDATAPATH():
    assert WRONGDATAPATH != ""

    