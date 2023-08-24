import unittest

def match_layers_with_paths(layer, path):
    return [p for item in layer for p in path if item in p]

class TestLayerPathMatching(unittest.TestCase):

    def test_matching_paths(self):
        layer = [
            "cross_attn",
            "self_attn"
        ]

        path = [
            "module.self_attn.v_prot",
            "module.self_cattn.k_prot",
            "module.cross_attn.q_prot"
        ]

        path_id = match_layers_with_paths(layer, path)
        expected_path_id = ['module.self_attn.v_prot', 'module.cross_attn.q_prot']

        self.assertCountEqual(path_id, expected_path_id)

if __name__ == '__main__':
    unittest.main()
