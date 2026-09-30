import { createStore } from "/js/AlpineStore.js";
import { callJsonApi } from "/js/api.js";
import { toastFrontendError } from "/components/notifications/notification-store.js";

export const store = createStore("autodreamStats", {
    data: null,
    loading: false,
    requestId: 0,

    async load(projectName = "", agentProfile = "") {
        const requestId = ++this.requestId;
        this.loading = true;
        this.data = null;
        try {
            const data = await callJsonApi("/plugins/autodream/stats", {
                project_name: projectName,
                agent_profile: agentProfile,
            });
            if (requestId === this.requestId) this.data = data;
        } catch (error) {
            if (requestId === this.requestId) {
                toastFrontendError(error?.message || "Could not load dream statistics", "AutoDream");
            }
        } finally {
            if (requestId === this.requestId) this.loading = false;
        }
    },

    get stats() { return this.data?.stats || {}; },
    get last() { return this.stats.last_run || {}; },
    get status() {
        if (this.loading) return "Loading…";
        if (!this.data) return "Unavailable";
        if (this.data.running) return "Dreaming";
        return ({ updated: "Memories updated", noop: "No changes needed", failed: "Last dream failed" })[this.last.status]
            || (this.data.last_dream_at ? "Waiting for next dream" : "Ready to dream");
    },
    number(value) { return this.data ? Number(value || 0).toLocaleString() : "—"; },
    date(value) { return value ? new Date(value).toLocaleString() : "No dreams recorded yet"; },
    duration(value) {
        if (value == null) return "—";
        const seconds = Math.round(value);
        return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    },
    cleanup() { this.requestId++; },
});
