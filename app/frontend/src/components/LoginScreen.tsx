import React from "react";
import type { FormEvent } from "react";
import { Loader2, LockKeyhole, LogIn } from "lucide-react";

interface LoginScreenProps {
  username: string;
  password: string;
  isSubmitting: boolean;
  errorMessage: string | null;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
}

export const LoginScreen: React.FC<LoginScreenProps> = ({
  username,
  password,
  isSubmitting,
  errorMessage,
  onSubmit,
  onUsernameChange,
  onPasswordChange,
}) => {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-medical-50 px-4 py-10">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),_transparent_34%),radial-gradient(circle_at_bottom_right,_rgba(15,23,42,0.14),_transparent_30%)]" />

      <div className="relative w-full max-w-[480px]">
        <section className="rounded-[2rem] border border-medical-200 bg-white p-8 shadow-xl md:p-10">
          <div className="mb-8">
            <div className="inline-flex rounded-2xl bg-medical-100 p-3 text-medical-800">
              <LockKeyhole size={22} />
            </div>
            <h2 className="mt-4 text-2xl font-semibold text-medical-900">Login</h2>
          </div>

          <form className="space-y-4" onSubmit={onSubmit}>
            <label className="block space-y-2 text-sm text-medical-700">
              <span className="font-medium">Username</span>
              <input
                value={username}
                onChange={(event) => onUsernameChange(event.target.value)}
                className="w-full rounded-2xl border border-medical-200 bg-medical-50 px-4 py-3 outline-none transition focus:border-accent"
                autoComplete="username"
                placeholder="Enter username"
              />
            </label>

            <label className="block space-y-2 text-sm text-medical-700">
              <span className="font-medium">Password</span>
              <input
                type="password"
                value={password}
                onChange={(event) => onPasswordChange(event.target.value)}
                className="w-full rounded-2xl border border-medical-200 bg-medical-50 px-4 py-3 outline-none transition focus:border-accent"
                autoComplete="current-password"
                placeholder="Enter password"
              />
            </label>

            {errorMessage && (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {errorMessage}
              </div>
            )}

            <button
              type="submit"
              disabled={isSubmitting}
              className="flex w-full items-center justify-center gap-2 rounded-2xl bg-medical-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-medical-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSubmitting ? <Loader2 size={16} className="animate-spin" /> : <LogIn size={16} />}
              Sign in
            </button>
          </form>
        </section>
      </div>
    </div>
  );
};