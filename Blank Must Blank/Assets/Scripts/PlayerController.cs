﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Vuforia;

public class PlayerController : MonoBehaviour {
    private CanvasController canvasController;
    private int cogs;

    private ShopController shopController;

    public List<Trap> traps = new List<Trap>();

    [System.Serializable]
    public class Trap {
        public enum Type {
            Spikes, Acid, Turret
        }

        public Type type;
        public int cost;
        public List<GameObject> targets;
        [HideInInspector] public int lastMarker;

        public Trap(Type type, int cost) {
            this.targets = new List<GameObject>();
            this.type = type;
            this.cost = cost;
            this.lastMarker = 0;
        }
    }

    private void Start() {
        this.canvasController = GameObject.FindGameObjectWithTag("Canvas").GetComponent<CanvasController>();
        this.shopController = GameObject.FindGameObjectWithTag("Shop").GetComponent<ShopController>();

        StartCoroutine(DisableImageTargets());
    }

    public void ShopTrap(Trap.Type trapType, bool isPurchase) {
        Trap trap = this.traps.Find(t => t.type == trapType);

        if (isPurchase && this.cogs < 0) {
            Debug.Log("Not enough money!");
            return;
        }

        if (isPurchase && trap.lastMarker < 5) {
            trap.lastMarker++;
            this.cogs -= trap.cost;
            trap.targets[trap.lastMarker - 1].GetComponent<ImageTargetBehaviour>().enabled = true;
        }

        if (!isPurchase && trap.lastMarker > 0) {
            trap.lastMarker--;
            this.cogs += trap.cost;
            trap.targets[trap.lastMarker].GetComponent<ImageTargetBehaviour>().enabled = false;
        }

        this.shopController.UpdateInventoryLabel(trap.type, trap.lastMarker);
    }

    private IEnumerator DisableImageTargets() {
        yield return new WaitForSeconds(0.1f);
        this.traps.ForEach(trap => trap.targets.ForEach(target => {
            target.GetComponent<ImageTargetBehaviour>().enabled = false;
        }));
    }

    public int GetCogs() {
        return this.cogs;
    }

    public void CollectCogs(int amount) {
        this.cogs += amount;
    }
}
