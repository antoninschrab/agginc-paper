import seaborn as sns

TEST_NAMES = ['L2 SechExp', 'L1 IMQ',
              'IMQ KSD', 'Gauss KSD',
              'Gauss FSSD-rand', 'Gauss FSSD-opt',
              'Gauss RFF', 'Cauchy RFF',
              'L1 IMQ (alpha)', 'L2 SechExp (alpha)']

ORDERED_TEST_NAMES = ['L2 SechExp', 'L2 SechExp (alpha)',
                      'L1 IMQ', 'L1 IMQ (alpha)', 'L1 IMQ (RBM)',
                      'Gauss KSD', 'IMQ KSD',
                      'Gauss FSSD-rand', 'Gauss FSSD-opt',
                      'Gauss RFF', 'Cauchy RFF']

def test_name_colors_dict():
    return dict(zip(TEST_NAMES, sns.color_palette(n_colors=len(TEST_NAMES))))
