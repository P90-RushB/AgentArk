// Executes the production viewer script with a small DOM, without browser packages.
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');

class Element {
    constructor() { this.children = []; this.className = ''; this._html = ''; this.textContent = ''; this.value = ''; }
    get innerHTML() { return this._html; }
    set innerHTML(value) { this._html = value; this.children = []; }
    appendChild(child) { this.children.push(child); }
    removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
    get firstChild() { return this.children[0]; }
}

async function main(input) {
    const elements = new Map();
    const getElement = id => {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
    };
    let stateLoaded = false;
    let connectedAfterState = false;
    let stream;
    const context = vm.createContext({
        document: {getElementById: getElement, createElement: () => new Element(), body: {scrollHeight: 0}},
        window: {scrollTo() {}},
        fetch: async () => ({json: async () => {
            stateLoaded = true;
            return {events: input.state_events || []};
        }}),
        EventSource: class {
            constructor() { connectedAfterState = stateLoaded; stream = this; }
        },
    });
    const script = input.html.match(/<script>([\s\S]*?)<\/script>/)[1];
    vm.runInContext(script, context);
    await new Promise(resolve => setImmediate(resolve));
    const snapshot = () => getElement('chat').children.map(row => {
        if (!input.compact) return {className: row.className, html: row.innerHTML};
        const hash = text => crypto.createHash('sha256').update(text).digest('hex');
        return {
            className: row.className,
            html_sha256: hash(row.innerHTML),
            images: [...row.innerHTML.matchAll(/<img src="([^"]+)"/g)].map(match => hash(match[1])),
        };
    });
    const snapshots = [];
    for (const [index, event] of (input.events || []).entries()) {
        if ((input.clear_at || []).includes(index)) getElement('clearEvents').onclick();
        stream.onmessage({data: JSON.stringify(event)});
        if (input.snapshots) snapshots.push(snapshot());
    }
    return {
        rows: snapshot(), snapshots, connectedAfterState,
        raw_count: getElement('eventCount').textContent,
        raw_request: getElement('rawRequestTrace').textContent,
        raw_response: getElement('rawResponseTrace').textContent,
    };
}

main(JSON.parse(fs.readFileSync(0, 'utf8'))).then(result => {
    process.stdout.write(JSON.stringify(result));
}).catch(error => { console.error(error); process.exitCode = 1; });
