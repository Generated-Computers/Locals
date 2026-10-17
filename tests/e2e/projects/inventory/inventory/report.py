from .store import Store


def tag_report(store: Store, tags) -> str:
    """One line per tag: the tag, the number of items with it and their value."""
    lines = []
    for tag in tags:
        items = store.find_by_tag(tag)
        lines.append(f"{tag}: {len(items)} items, {sum(i.value for i in items):.2f}")
    return "\n".join(lines)


def cheapest_with_tag(store: Store, tag: str):
    items = store.find_by_tag(tag)
    return min(items, key=lambda i: i.price) if items else None
