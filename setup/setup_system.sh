#!/bin/bash
#!/bin/bash

set_var() {
    var_name=$1
    var_val=$2
    # Check if the variable already exists in the setup/.vars file
    if grep -q "^${var_name}=" setup/.vars; then
        # Update the variable's value
        sed -i "s/^${var_name}=.*/${var_name}=${var_val}/" setup/.vars
    else
        # Append the variable to the file
        echo "${var_name}=${var_val}" >>setup/.vars
    fi
}

init() {
    ENV_NAME=${1:-masterarbeit}  # Default to 'masterarbeit' if not provided
    USE_GPU=${2:-false}          # Default to false if not provided

    # Check if the conda environment exists, if not create it
    if ! conda env list | grep -q "^$ENV_NAME\s"; then
        conda create -n "$ENV_NAME" python=3.11.5 -y
    fi

    # Activate the conda environment
    conda activate "$ENV_NAME"

    # Install the appropriate dependencies based on USE_GPU
    if [ "$USE_GPU" = true ]; then
        $HOME/anaconda3/envs/$ENV_NAME/bin/python -m pip install -r setup/requirements_gpu.txt
    else
        $HOME/anaconda3/envs/$ENV_NAME/bin/python -m pip install -r setup/requirements_cpu.txt
    fi

    # Set or update system variables
    set_var "ENV_NAME" "$ENV_NAME"
    set_var "USE_GPU" "$USE_GPU"

    # Reload setup/.vars to apply changes
    if [ -f setup/.vars ]; then
        source setup/.vars
    fi
}

# Interactive input function
ask_for_input() {
    if [ -z "$ENV_NAME" ]; then
        read -p "Enter the environment name (default: masterarbeit): " ENV_NAME
        ENV_NAME=${ENV_NAME:-masterarbeit}  # Set to default if empty
    fi

    if [ -z "$USE_GPU" ]; then
        read -p "Do you want to use GPU? (y/[n]): " gpu_choice
        if [[ "$gpu_choice" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            USE_GPU=true
        else
            USE_GPU=false
        fi
    fi
}

# -----------------------------------------------
# Main loop for argument parsing

ENV_NAME=""
USE_GPU=""

if [ -f setup/.vars ]; then
    source setup/.vars
else
    touch setup/.vars
fi

# Ask for input interactively if arguments are not provided
ask_for_input

# Call the init function with arguments
init "$ENV_NAME" "$USE_GPU"
