# The store



The store is a key element that acts as a container for the variables of the various components and is dynamically updated as the system runs. It also serves as a means for different components to have access to variables that were created by other components when overwritten hooks were called.

The store is a `SimpleNameSpace` object, which allows for values to be stored as attributes on the fly without having to be explicitly created first.
### The store's initialization
The store is initially generated in the initialisation of the `Builder` component, after which each distributed process receives its own deep copy of the initial store to be modified as hooks are called by that process. For example the `trainer` process will add `grad_fn`, which is a function to be used for computing the gradient based on a loss function, to its store when the `on_training_loss_fns` hook is called.
