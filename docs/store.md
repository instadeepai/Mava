# The store

> 🚧 **Note:** This only applies to the callback redesign of Mava.

The store is a key element that acts as a container for assigning the variables of the various components dynamically and that vary as the system operates.

The store is a `SimpleNameSpace` instance, which allows users to store values as attributes without having to create their own class, which is usually required to be almost empty.

### The store's initialization
The store is initially generated in the initialisation of the `builder` component, then subsequently in the distribution process, the other components receive a copy (deep copy) of the initial store and can even add extra variables particular to that component to their own copy (e.g the `executor` component will add `executor_id` to it store).
