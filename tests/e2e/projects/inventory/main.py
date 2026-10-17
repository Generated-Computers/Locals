from inventory import Item, Store
from inventory.report import tag_report

store = Store()
store.add(Item("apple", 0.5, 10, ["fruit", "fresh"]))
store.add(Item("pear", 0.75, 4, ["fruit"]))
store.add(Item("soap", 2.0, 3, ["home"]))
print(tag_report(store, ["fruit", "home"]))
print("fresh:", [i.name for i in store.find_by_tag("fresh")])
