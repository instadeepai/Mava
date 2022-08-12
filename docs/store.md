# The store
The store is a key element that acts as a container for assigning the variables of the various components dynamically and that vary as the system operates.

The store is a `SimpleNameSpace` instance, which allows users to store values as attributes without having to create their own class, which is usually required to be almost empty.

## The initialise of the store
The store is initially generated in the initialisation of the `builder` component, then subsequently in the distribution process, the other components receive a copy (deep copy) of the initial store and can even add extra variables particular to that component to their own copy.