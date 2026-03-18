""" Some useful constants used throughout codebase """

DATASETS = {"cyp2c9": {"ds_name": "cyp2c9",
                      "ds_dir": "data/cyp2c9",
                      "ds_fn": "../data/cyp2c9/cyp2c9_dg.tsv",
                      "wt_aa": "MDSLVVLVLCLSCLLLLSLWRQSSGRGKLPPGPTPLPVIGNILQIGIKDISKSLTNLSKV"
                               "YGPVFTLYFGLKPIVVLHGYEAVKEALIDLGEEFSGRGIFPLAERANRGFGIVFSNGKKW"
                               "KEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICS"
                               "IIFHKRFDYKDQQFLNLMEKLNENIKILSSPWIQICNNFSPIIDYFPGTHNKLLKNVAFM"
                               "KSYILEKVKEHQESMDMNNPQDFIDCFLMKMEKEKHNQPSEFTIESLENTAVDLFGAGTE"
                               "TTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRSHMPYTDAVVHEVQRYID"
                               "LLPTSLPHAVTCDIKFRNYLIPKGTTILISLTSVLHDNKEFPNPEMFDPHHFLDEGGNFK"
                               "KSKYFMPFSAGKRICVGEALAGMELFLFLTSILQNFNLKSLVDPKNLDTTPVVNGFASVP"
                               "PFYQLCFIPV",
                      "wt_ofs": 490,
                      "pdb_fn": "../data/cyp2c9/AF-P11712-F1-model_v4.pdb"}}

# list of chars that can be encountered in any sequence
CHARS = ["X", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# number of chars
NUM_CHARS = 21  # len(CHARS)

# dictionary mapping chars->int
C2I_MAPPING = {c: i for i, c in enumerate(CHARS)}
