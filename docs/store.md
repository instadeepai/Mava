# The store

> 🚧 **Note:** This only applies to the callback redesign of Mava.

The store is a key element that acts as a container for the variables of the various components and is dynamically updated as the system runs. It also serves as a means for different components to have access to variables that were created by other components when overwritten hooks were called.

The store is a `SimpleNameSpace` object, which allows for values to be stored as attributes on the fly without having to be explicit created first.
### The store's initialization
The store is initially generated in the initialisation of the `Builder` component, after which each distributed process receives its own copy (deep copy) of the initial store to be modified as hooks are called by that process. For example the `executor` process will add `executor_id` to its store.
