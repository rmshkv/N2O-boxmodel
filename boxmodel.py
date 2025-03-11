import xarray as xr
import yaml

def initialize(path_to_param_yaml):
    """
    Initializes a state array to input into the model from a set of parameters.

    Args: 
    
        path_to_param_yml: path to the parameters yaml file (str)

    Returns: 
    
        state_array: state array for initial time step (xarray Dataset)
    """

    ### read in yaml file with all parameters

    with open(path_to_param_yaml) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    ### retrieve initial values for concentrations and isotopes

    N2O_14_top_init = params['N2O_top_init']
    N2O_15_top_init = params['N2O_top_init'] * params['R15_N2O_top_init']
    
    NO3_14_top_init = params['NO3_top_init']
    NO3_15_top_init = params['NO3_top_init'] * params['R15_NO3_top_init']

    N2O_14_bottom_init = params['N2O_bottom_init']
    N2O_15_bottom_init = params['N2O_bottom_init'] * params['R15_N2O_bottom_init']
    
    NO3_14_bottom_init = params['NO3_bottom_init']
    NO3_15_bottom_init = params['NO3_bottom_init'] * params['R15_NO3_bottom_init']

    ### create state array

    # add units and metadata to this later

    state_array = xr.Dataset(
    {
        "NO3_14": (["depth", "time"], [[NO3_14_top_init], [NO3_14_bottom_init]], {"units": "uM"}),
        "NO3_15": (["depth", "time"], [[NO3_15_top_init], [NO3_15_bottom_init]], {"units": "uM"}),
        "N2O_14": (["depth", "time"], [[N2O_14_top_init], [N2O_14_bottom_init]], {"units": "uM"}),
        "N2O_15": (["depth", "time"], [[N2O_15_top_init], [N2O_15_bottom_init]], {"units": "uM"}),
    },
    coords = {"depth": ("depth", [0,-1]),
    
        "time": ("time", [0], {"units": "days"})
    },
    )

    return state_array

def step_forward(path_to_param_yaml, state_array, dt):
    """
    Steps the model forward once.

    Args: 
    
        path_to_param_yml: path to the parameters yaml file (str)

        state_array: state array for the previous timestep (xarray Dataset)

        dt: timestep size (float)

    Returns:

        new_state_array: state array with the new timestep appended (xarray Dataset)

    """

    ### read in yaml file with all parameters

    with open(path_to_param_yaml) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    ### add to state array

    top_array_prev = state_array.sel({"depth": 0}).isel({"time":-1})
    bottom_array_prev = state_array.sel({"depth": -1}).isel({"time":-1})

    next_N2O_14_top = (top_array_prev.N2O_14
                       + dt**2 * params['F_NH4_in'] * params['k_nitrif_N2O']
                       + bottom_array_prev.N2O_14 * dt * params['k_mixing']
                       - top_array_prev.N2O_14 * dt * params['k_mixing']
                       )

    next_N2O_15_top = (top_array_prev.N2O_15 
                       + dt**2 * params['F_NH4_in'] * params['R15_NH4_in'] * params['k_nitrif_N2O'] * params['alpha_nitrif_N2O'] 
                       + bottom_array_prev.N2O_15 * dt * params['k_mixing']
                       - top_array_prev.N2O_15 * dt * params['k_mixing']
                       )
    
    next_NO3_14_top = (top_array_prev.NO3_14 
                       + dt**2 * params['F_NH4_in'] *  params['k_nitrif']
                       + bottom_array_prev.NO3_14 * dt * params['k_mixing']
                       - top_array_prev.NO3_14 * dt * params['k_mixing']
                       )

    next_NO3_15_top = (top_array_prev.NO3_15 
                       + dt**2 * params['F_NH4_in'] * params['R15_NH4_in'] * params['k_nitrif'] * params['alpha_nitrif'] 
                       + bottom_array_prev.NO3_15 * dt * params['k_mixing']
                       - top_array_prev.NO3_15 * dt * params['k_mixing']
                       )

        ### later: update denitrification to take in total nitrate rather than just 14NO3

    next_N2O_14_bottom = (bottom_array_prev.N2O_14
                          + dt**2 * params['F_NO2_in'] * params['k_denitrif_1']
                          + dt * bottom_array_prev.NO3_14 * params['k_denitrif_2'] 
                          + top_array_prev.N2O_14 * dt * params['k_mixing']
                          - dt * bottom_array_prev.N2O_14 * params['k_denitrif_3'] 
                          - bottom_array_prev.N2O_14 * dt * params['k_mixing']
                          )

    next_N2O_15_bottom = (bottom_array_prev.N2O_15
                          + dt**2 * params['F_NO2_in'] * params['R15_NO2_in'] * params['k_denitrif_1'] * params['alpha_denitrif_1']
                          + dt * bottom_array_prev.NO3_15 * params['k_denitrif_2'] * params['alpha_denitrif_2']
                          + top_array_prev.N2O_15 * dt * params['k_mixing']
                          - dt * bottom_array_prev.N2O_15 * params['k_denitrif_3'] * params['alpha_denitrif_3'] 
                          - bottom_array_prev.N2O_15 * dt * params['k_mixing']
                          )
    
    next_NO3_14_bottom = (bottom_array_prev.NO3_14
                          + dt * params['F_NO3_in'] 
                          + top_array_prev.NO3_14 * dt * params['k_mixing']
                          - dt * bottom_array_prev.NO3_14 * params['k_denitrif_2']
                          - bottom_array_prev.NO3_14 * dt * params['k_mixing']
                          )


    next_NO3_15_bottom = (bottom_array_prev.NO3_15 
                          + dt * params['F_NO3_in'] * params['R15_NO3_in']
                          + top_array_prev.NO3_15 * dt * params['k_mixing']
                          - dt * bottom_array_prev.NO3_15 * params['k_denitrif_2'] * (1/params['alpha_denitrif_2']) # make sure to include this
                          - bottom_array_prev.NO3_15 * dt * params['k_mixing']
                          )
            
    ### create an array for this new timestep
    
    next_time = state_array.isel({"time":-1}).time + dt

    next_state_array = xr.Dataset(
    {
        "NO3_14": (["depth", "time"], [[next_NO3_14_top], [next_NO3_14_bottom]]),
        "NO3_15": (["depth", "time"], [[next_NO3_15_top], [next_NO3_15_bottom]]),
        "N2O_14": (["depth", "time"], [[next_N2O_14_top], [next_N2O_14_bottom]]),
        "N2O_15": (["depth", "time"], [[next_N2O_15_top], [next_N2O_15_bottom]]),
    },
    coords={
        "depth": [0,-1],
        "time": [next_time],
    },
    )

    ### quick failsafe to make negative concentrations into zero:

    next_state_array = next_state_array.where(next_state_array>=0, 0)

    ### concatenate the new timestep to the previous state array
    
    new_state_array = xr.concat([state_array, next_state_array], "time")

    return new_state_array

def run_model(path_to_param_yaml, dt, total_dt_count):
    """
    Runs the model for `total_dt_count` timesteps of size `dt`.

    Args: 
    
        path_to_param_yml: path to the parameters yaml file (str)

        dt: timestep size (float)

        total_dt_count: number of timesteps of size dt to run for (int)

    Returns:

        state_array: state array for the entire model run (xarray Dataset)
    """
    
    prev_state_array = initialize(path_to_param_yaml)

    for i in range(total_dt_count - 1): # subtract 1 because we calculate the first time step separately

        state_array = step_forward(path_to_param_yaml, prev_state_array, dt)
        
        prev_state_array = state_array

    return state_array