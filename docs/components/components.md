# Mava Components

The callback design centers around combining various component classes. Each components will overwrite relevant hooks which form the system's functionality. Each component class has a related component config class where its associated hyperparameter defaults may be defined. The hyperparameters that are defined in a component config class are also able to be overwritten at build time by passing in updated values to the `.build` method of a given system.
