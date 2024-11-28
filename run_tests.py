import unittest

def run_all_tests():

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)


if __name__ == '__main__':
    run_all_tests()
