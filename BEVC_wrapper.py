print("Hello world")

class modelWrapperBEVC():
    def __init__(self) -> None:
        # Importing modules
        from os import path
        from pybevc import PyBEVC
        from pybevc.test.data import test_data_path
        from pybevc.file_io import read_operation
        import matplotlib.pyplot as plt
        import hvplot.xarray

        # Instanciating a PyBEVC object
        self.bevc = PyBEVC()

        # Setting inputs manually 
        self.bevc.U0 = 8.0
        self.bevc.TSR = 7
        self.bevc.flag_a_CT = 2

        # Setting inputs from files
        self.bevc.from_windIO(path.join(test_data_path, "IEA-3.4-130-RWT.yaml"));  # windIO file

    def iterate(self, pos) -> dict:
        """
        pos: list of node coordinates in the blade frame of reference. The nodes are located in the elastic centre.

        Return:
            loads_dict: dictionary of arrays with the loads in the nodes.
        """
        # Setting the new blade centre line
        # Get blade half chord
        c2_dict = {'x':[0],'y':[0],'z':[0]}
        self.bevc.from_dict(c2_dict)

        # Running the BEVC solver and returning an Xarray DataSet
        res = self.bevc.run()

        # Setting input from bevc object
        self.bevc.add_inp_to_out(res)
        # Setting computed values (e.g. CT, CP, CLT, CLP)
        res.bevc_com_vals.set_all()

        loads_dict = {
            'fx': res['fx_b'].data,
            'fy': res['fy_b'].data,
            'fz': res['fz_b'].data,
            'mx': res['sec_mx_b'].data,
            'my': res['sec_my_b'].data,
            'mz': res['sec_mz_b'].data,
            }

        return loads_dict