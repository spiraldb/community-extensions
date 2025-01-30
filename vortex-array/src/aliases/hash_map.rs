pub type DefaultHashBuilder = hashbrown::DefaultHashBuilder;
pub type RandomState = hashbrown::DefaultHashBuilder;
pub type HashMap<K, V, S = DefaultHashBuilder> = hashbrown::HashMap<K, V, S>;
pub type Entry<'a, K, V, S> = hashbrown::hash_map::Entry<'a, K, V, S>;
pub type IntoIter<K, V> = hashbrown::hash_map::IntoIter<K, V>;
pub type HashTable<T> = hashbrown::HashTable<T>;
