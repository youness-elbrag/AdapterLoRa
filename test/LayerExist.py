import unittest

def LayerType(layer):
    layers = ["nn.Linear" , "nn.Embedding", "nn.Conv1d","nn.Conv2d"]
    if layers.__contains__(layer) == False:
        return f"{layer} not support Please Visit \n Docs to list correct Layer support"
    return True

class TestCaseExitLayer(unittest.TestCase):

    def ExitLayer(self):
        layer = "nn.Linear"
        ExpectOut = True
        Result = LayerType(layer)
        self.assertTrue(Result,ExpectOut)

if __name__ == "__main__":
    unittest.main()






