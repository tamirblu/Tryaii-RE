/**
 * User priority system for model selection.
 *
 * Priorities let users express what matters to them (quality, cost, speed)
 * on a 1-5 scale. These get transformed into weights that influence scoring.
 */

export interface PrioritiesData {
  quality: number;
  cost: number;
  speed: number;
}

export class Priorities {
  /**
   * Each value is on a 1-5 scale:
   *   1 = don't care about this dimension
   *   3 = balanced (default)
   *   5 = this is critical
   */
  readonly quality: number;
  readonly cost: number;
  readonly speed: number;

  constructor(quality = 3, cost = 3, speed = 3) {
    this.quality = Math.max(1, Math.min(5, Math.round(quality)));
    this.cost = Math.max(1, Math.min(5, Math.round(cost)));
    this.speed = Math.max(1, Math.min(5, Math.round(speed)));
  }

  /** Quality weight: 0.3 to 1.2 (always has baseline influence). */
  get qualityWeight(): number {
    return 0.3 + (this.quality / 5) * 0.9;
  }

  /** Cost weight: 0.1 to 1.0 (can be fully suppressed). */
  get costWeight(): number {
    return 0.1 + (this.cost / 5) * 0.9;
  }

  /** Speed weight: 0.1 to 1.0 (can be fully suppressed). */
  get speedWeight(): number {
    return 0.1 + (this.speed / 5) * 0.9;
  }

  toDict(): PrioritiesData {
    return { quality: this.quality, cost: this.cost, speed: this.speed };
  }

  static fromDict(d: Partial<PrioritiesData>): Priorities {
    return new Priorities(d.quality ?? 3, d.cost ?? 3, d.speed ?? 3);
  }

  /** Preset: maximize quality, ignore cost and speed. */
  static performance(): Priorities {
    return new Priorities(5, 1, 1);
  }

  /** Preset: minimize cost, moderate quality. */
  static budget(): Priorities {
    return new Priorities(2, 5, 3);
  }

  /** Preset: fastest response, moderate quality. */
  static fast(): Priorities {
    return new Priorities(2, 3, 5);
  }

  /** Preset: balanced across all dimensions. */
  static balanced(): Priorities {
    return new Priorities(3, 3, 3);
  }
}

export const DEFAULT_PRIORITIES = new Priorities(3, 3, 3);
