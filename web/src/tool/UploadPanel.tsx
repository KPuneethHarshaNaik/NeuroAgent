import { useRef, useState } from "react";

type Props = {
  busy: boolean;
  status: string;
  onSubmit: (file: File) => void;
};

/** Same contract as the vanilla form: multipart POST to /live-jobs, field name "file". */
export function UploadPanel({ busy, status, onSubmit }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const accept = (file: File | undefined) => {
    if (!file) return;
    setName(file.name);
    onSubmit(file);
  };

  return (
    <section className="surface" aria-labelledby="upload-title">
      <div className="border-b border-line px-4 py-3">
        <p className="label-micro">Start a review</p>
        <h2 id="upload-title" className="mt-1 text-[17px] font-medium tracking-tight">
          Open an EEG recording
        </h2>
      </div>
      <div className="px-4 py-4">
        <p className="text-[13px] leading-relaxed text-muted">
          An event-marked <span className="num">.edf</span>, <span className="num">.fif</span> or{" "}
          <span className="num">.bdf</span> recording. The upload is stored unchanged and hashed before any stage reads
          it.
        </p>

        <label
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            accept(event.dataTransfer.files[0]);
          }}
          className={`mt-4 flex cursor-pointer items-center justify-between gap-3 border border-dashed px-3 py-3 transition-colors ${
            dragging ? "border-accent bg-accent/5" : "border-line hover:border-muted"
          }`}
          data-testid="upload-drop"
        >
          <input
            ref={inputRef}
            type="file"
            name="file"
            accept=".fif,.edf,.bdf"
            className="sr-only"
            disabled={busy}
            data-testid="upload-input"
            onChange={(event) => accept(event.target.files?.[0])}
          />
          <span className="truncate text-[13px]">{name ?? "Choose recording or drop it here"}</span>
          <span className="num shrink-0 text-[11px] text-muted">{busy ? "working…" : ".edf / .fif / .bdf"}</span>
        </label>

        <button
          type="button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          className="mt-3 w-full border border-accent bg-accent px-4 py-2.5 text-[13.5px] font-medium text-white transition-colors hover:bg-[#c42136] disabled:cursor-not-allowed disabled:opacity-50"
          data-testid="upload-submit"
        >
          {busy ? "Analysing…" : "Start analysis"}
        </button>

        <p className="mt-3 min-h-[18px] text-[12px] text-muted" role="status" data-testid="upload-status">
          {status || "No recording loaded."}
        </p>
      </div>
    </section>
  );
}
